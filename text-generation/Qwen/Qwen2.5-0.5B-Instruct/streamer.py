import time
import json
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from queue import Queue
from environment import MODEL_LOGGING
from adaptation import (
    ComplianceFormat,
    hf_token_logprobs_to_openai_format,
    coallesce_infinity_to_none,
    token_to_bytes,
    generate_id,
)

from transformers.generation.utils import GenerationMixin

# NOTE for monkey patching
OG__SAMPLE = GenerationMixin._sample


@dataclass
class TokenResult:
    token: str
    logprobs: dict = None
    text_offset: int = None
    is_first_token: bool = False
    is_last_token: bool = False


class SingleTokenStreamerVllm(BaseStreamer):
    def __init__(self):
        self.text_queue = Queue()

    def put(self, value: str):
        self.text_queue.put(value)

    def end(self):
        """If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is None:
            raise StopIteration()
        else:
            return value


class SingleTokenStreamer(BaseStreamer):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        skip_prompt: bool = False,
        logprobs: bool = False,
        top_logprobs=5,
        compliance_format: ComplianceFormat = ComplianceFormat.DEFAULT,
        model_id: str = None,
        **decode_kwargs,
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs
        self.text_queue = Queue()
        self.prev_token: TokenResult = None
        self.prompt_skipped = False
        self.stop_signal = None
        self.token_buffer = []
        self.all_buffer = []
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.compliance_format = compliance_format
        # required for OAI compliance
        self.model_id = model_id
        self.id = generate_id()
        # part of the monkey patch to get the logit scores when .put() is called
        self.logits_processor = None
        self.next_token_scores = None
        self.monkey_patch_transformers_text_generation_sampler()

    def reset(self):
        self.text_queue = Queue()
        self.pending_tokens = []
        self.token_buffer = []
        self.prompt_skipped = False
        GenerationMixin._sample = OG__SAMPLE

    # NOTE the purpose of this is to set self.logits_processor so that when .put() is called
    # we have access to the logit scores so we can compute logprobs if they are enabled
    def monkey_patch_transformers_text_generation_sampler(streamer_self):
        if not streamer_self.logprobs:
            return

        def patched__sample(self, *args, **kwargs):
            # NOTE it may be better to just do this on:
            # probs = nn.functional.softmax(next_token_scores, dim=-1)
            # and pass the probs in directly as we do a second softmax operation
            streamer_self.logits_processor = kwargs.get("logits_processor")

            def wrapped_logits_processor(*args, **kwargs):
                next_token_scores = streamer_self.logits_processor(*args, **kwargs)

                streamer_self.next_token_scores = next_token_scores

                return next_token_scores

            # NOTE this is only safe because the kwargs ref changes everytime _sample() is called
            kwargs["logits_processor"] = wrapped_logits_processor

            return OG__SAMPLE(self, *args, **kwargs)

        GenerationMixin._sample = patched__sample

    def put(self, value):
        # NOTE part of the monkey patch
        logit_scores = self.next_token_scores

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        value_as_list = value.tolist()

        # include all tokens if it's the prompt
        if self.skip_prompt and not self.prompt_skipped:
            self.token_buffer = value_as_list
            # keep a separate instance of the list, these needs to be unique objects
            self.all_buffer = list(value_as_list)
            self.prompt_skipped = True

            # we skip to the next iteration, our diff is useless in this case
            return

        token = value_as_list[0]
        self.token_buffer.append(token)
        self.all_buffer.append(token)

        text: str = self.tokenizer.decode(self.token_buffer, **self.decode_kwargs)

        if MODEL_LOGGING:
            print(text)

        # we hit a nonsense piece of text, so we wait for the next token
        if "�" in text:
            return

        # this guarantees that regardless of the model, if tokens are not interpreted in isolation that " "'s and other characters are properly added
        # we compare the text for the previous set of tokens against the new set of tokens.
        # e.g. tokens [1,2] could map to 1 => "A", 2 => "car", if you decode one by one, you would get the string "Acar"
        # if you do them together, you get "A car"
        # so all we're doing is taking "A" and "A car" and slicing off "A" from "A car" to give us " car"
        prev_text = self.tokenizer.decode(self.all_buffer[:-1], **self.decode_kwargs)
        full_text = self.tokenizer.decode(self.all_buffer, **self.decode_kwargs)

        text_offset = len(prev_text)

        diff_text = full_text[text_offset:]

        logprobs = (
            None
            if not self.logprobs
            else hf_token_logprobs_to_openai_format(
                token_id=token,
                logit_scores=logit_scores,
                tokenizer=self.tokenizer,
                top_k=self.top_logprobs,
            )
        )

        self.token_buffer = []

        token_result = TokenResult(
            token=diff_text, logprobs=logprobs, text_offset=text_offset
        )

        # this allows us to treat the last token differently because it will never be put into the queue here
        # it has to be "manually" extracted via end()
        if self.prev_token:
            self.text_queue.put(self.prev_token)
        else:
            token_result.is_first_token = True

        self.prev_token = token_result

    def __iter__(self):
        return self

    def __next__(self) -> str:
        # this needs to wait for a signal that a buffer has been filled with at least one before it triggers
        # we then need end to get the last token
        value = self.text_queue.get()
        return self.handle_output_format(value)

    def end(self):
        # add the last token value and mark it as the last token
        token_value = self.prev_token

        if token_value:
            token_value.is_last_token = True
            self.text_queue.put(token_value)

        # an extra stop before we actually stop to allow the iterator to get a last line if the compliance format calls for it
        self.text_queue.put(self.compliance_format)

        self.text_queue.put(self.stop_signal)

        # NOTE always restore sample to its original state
        GenerationMixin._sample = OG__SAMPLE

    def handle_output_format(self, value: TokenResult):
        if value is self.stop_signal:
            raise StopIteration()

        # this is a little weird, if we're streaming openai style, we put the compliance format in the stream before the
        # None as our last item to allow us to add the [DONE]
        if isinstance(value, ComplianceFormat):
            if value == ComplianceFormat.DEFAULT:
                raise StopIteration()

            if value.is_openai_format():
                # Final SSE line per OpenAI spec
                return f"data: [DONE]\n\n"

        value: TokenResult

        if self.compliance_format == ComplianceFormat.DEFAULT:
            return value.token

        finish_reason = "stop" if value.is_last_token else None

        if self.compliance_format == ComplianceFormat.OPEN_AI_COMPLETIONS:
            logprobs = (
                {
                    "tokens": [value.token],
                    "token_logprobs": [
                        coallesce_infinity_to_none(value.logprobs["logprob"])
                    ],
                    "top_logprobs": value.logprobs["top_logprobs"],
                    # TODO make this correct
                    "text_offset": [value.text_offset],
                }
                if value.logprobs is not None
                else None
            )
            payload = {
                "id": f"cmpl-{self.id}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": value.token,
                        "logprobs": logprobs,
                        "finish_reason": finish_reason,
                        "stop_reason": None,
                    }
                ],
                "usage": None,
            }

            # Wrap as SSE line
            return f"data: {json.dumps(payload)}\n\n"

        if self.compliance_format == ComplianceFormat.OPEN_AI_CHAT_COMPLETIONS:
            logprobs = (
                {
                    "content": [
                        {
                            "token": value.token,
                            # JSON does not understand the value -inf or inf
                            "logprob": coallesce_infinity_to_none(
                                value.logprobs["logprob"]
                            ),
                            "bytes": token_to_bytes(value.token),
                            "top_logprobs": value.logprobs["top_logprobs"],
                        }
                    ]
                }
                if value.logprobs is not None
                else None
            )
            payload = {
                "id": f"chatcmpl-{self.id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            #
                            "content": value.token
                        },
                        "logprobs": None if value.is_first_token else logprobs,
                        "finish_reason": finish_reason,
                    },
                ],
            }

            # Wrap as SSE line
            return f"data: {json.dumps(payload)}\n\n"
