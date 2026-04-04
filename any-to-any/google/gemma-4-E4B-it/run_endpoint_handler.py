import copy
import time
from flask import jsonify, Response

from environment import MODEL_ID
from utils import model_run_generator
from model import model_run, pipe
from model_loader import LOADED_ON_VLLM
import numpy as np
from adaptation import (
    ComplianceFormat,
    get_and_delete_adaptation_kwargs,
    hf_sequence_logprobs_to_openai_completions_format,
    hf_sequence_logprobs_to_openai_chat_completions_format,
    generate_openai_id,
    should_conform_to_input_expectations,
)
from contextlib import contextmanager


# NOTE this needs to not deep copy the kwargs
og_processor_merge_kwargs = pipe.processor._merge_kwargs


def patched_processor_merge_kwargs(*args, **kwargs):
    with patch_deepcopy():
        return og_processor_merge_kwargs(*args, **kwargs)


@contextmanager
def patch_deepcopy():
    og_deepcopy = copy.deepcopy

    def patched_deep_copy(kwargs: dict, memo=None):
        streamer = kwargs.get("streamer")

        if streamer:
            del kwargs["streamer"]

        copied_kwargs = og_deepcopy(kwargs)

        if streamer:
            copied_kwargs["streamer"] = streamer
        return copied_kwargs

    copy.deepcopy = patched_deep_copy

    try:
        yield
    finally:
        copy.deepcopy = og_deepcopy


# NOTE this is a monkey patch to allow streaming to work
# NOTE we may want to deepcopy() the model_inputs and **kwargs in the event the deletions we
# perform mutate anything, particularly model_inputs

og_forward = pipe._forward


def patched_forward(self, model_inputs, **kwargs):
    del model_inputs["text"]

    streamer = kwargs["generate_kwargs"].get("streamer")
    del kwargs["generate_kwargs"]

    return self.model.generate(
        **model_inputs,
        streamer=streamer,
        **kwargs,
    )


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    messages = request.json["text"]

    if "temperature" in params:
        params["temperature"] = max(params["temperature"], 0.01)

    adapt_messages(messages)

    stream = request.json.get("stream", False)

    if not LOADED_ON_VLLM:
        if stream:
            pipe._forward = patched_forward.__get__(pipe, type(pipe))

    try:
        adaptation_kwargs = get_and_delete_adaptation_kwargs(params)

        compliance_format: ComplianceFormat = adaptation_kwargs["compliance_format"]

        should_conform_to_input_expectations(compliance_format, messages)

        pipeline_kwargs = {
            **params,
            # these add props to the returned model output dict that are for computing logprobs and token usage
            "output_scores": True,
            "return_dict_in_generate": True,
        }

        vllm_kwargs = {**params, **adaptation_kwargs}

        params = vllm_kwargs if LOADED_ON_VLLM else pipeline_kwargs

        if stream:
            output_generator = model_run_generator(
                messages,
                params=params,
                adaptation_kwargs=adaptation_kwargs,
            )

            return Response(
                output_generator(),
                content_type="text/event-stream; charset=utf-8",
            )

        if LOADED_ON_VLLM:
            model_output = model_run(
                messages,
                params,
            )

            if compliance_format.is_openai_format():
                return jsonify(model_output)

            return jsonify({"output": model_output})

        should_get_logprobs = adaptation_kwargs["logprobs"]

        inputs = pipe.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=params.get("enable_thinking", False),
        )

        inputs = {k: v.to(pipe.model.device.type) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        outputs = pipe.model.generate(**inputs, **params)

        sequence = outputs["sequences"][0]

        response = pipe.processor.decode(
            # NOTE this may do something unexpected to thinking tokens, may need to be False, but if set to False
            # you will get the formatting characters
            sequence[input_len:],
            skip_special_tokens=True,
        )

        # Parse output
        model_output = pipe.processor.parse_response(response)

        model_output = {"generated_text": [model_output]}

        total_tokens = len(sequence)

        usage = {
            "prompt_tokens": input_len,
            "total_tokens": total_tokens,
            "completion_tokens": total_tokens - input_len,
            "prompt_tokens_details": None,
        }

        logprobs = (
            get_logprobs(
                adaptation_kwargs,
                messages,
                {"sequence": sequence, "scores": outputs["scores"]},
            )
            if should_get_logprobs
            else None
        )

        model_output = clean_special_floats(model_output)

        if compliance_format == ComplianceFormat.OPEN_AI_COMPLETIONS:
            return jsonify(
                {
                    "id": generate_openai_id("cmpl"),
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "text": model_output["generated_text"],
                            "logprobs": logprobs,
                            # TODO could benefit from knowing difference between length and stop
                            "finish_reason": "stop",
                            "stop_reason": None,
                        }
                    ],
                    "usage": usage,
                }
            )

        if compliance_format == ComplianceFormat.OPEN_AI_CHAT_COMPLETIONS:
            last_message = model_output["generated_text"][-1]
            return jsonify(
                {
                    "id": generate_openai_id("chatcmpl"),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "message": {
                                # these by default are included in the response, if the model returns them, these will be overridden to a new value
                                "reasoning_content": None,
                                "tool_calls": [],
                                **last_message,
                                # in the event the model does not return "assistant" for role, we override it here
                                "role": "assistant",
                            },
                            "logprobs": logprobs,
                        }
                    ],
                    # TODO could benefit from knowing difference between length and stop
                    "finish_reason": "stop",
                    "stop_reason": None,
                    "usage": usage,
                }
            )

        # clean up the output
        if "sequence" in model_output:
            del model_output["sequence"]

        if "scores" in model_output:
            del model_output["scores"]

        model_output["logprobs"] = logprobs
        model_output["usage"] = usage

        return jsonify({"output": [model_output]})
        pass
    finally:
        pipe._forward = og_forward


def adapt_messages(messages):
    for message in messages:
        if isinstance(message["content"], str):
            message["content"] = [{"type": "text", "text": message["content"]}]


def clean_special_floats(data):
    """Recursively replace NaN, Infinity, and -Infinity with None."""
    if isinstance(data, dict):
        return {k: clean_special_floats(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_special_floats(v) for v in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None
    return data


def get_logprobs(
    adaptation_kwargs: dict,
    messages,
    model_output,
):
    top_logprobs = adaptation_kwargs["top_logprobs"]

    sequence = model_output["sequence"]
    scores = model_output["scores"]

    if isinstance(messages, str):
        return hf_sequence_logprobs_to_openai_completions_format(
            sequence, scores, pipe.tokenizer, top_logprobs
        )

    return hf_sequence_logprobs_to_openai_chat_completions_format(
        sequence, scores, pipe.tokenizer, top_logprobs
    )
