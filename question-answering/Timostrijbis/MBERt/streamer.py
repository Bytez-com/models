from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from queue import Queue
from environment import MODEL_LOGGING


class SingleTokenStreamer(BaseStreamer):

    def __init__(
        self, tokenizer: AutoTokenizer, skip_prompt: bool = False, **decode_kwargs
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs
        self.text_queue = Queue()
        self.next_tokens_are_prompt = True
        self.stop_signal = None
        self.token_buffer = []
        self.all_buffer = []

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        value_as_list = value.tolist()

        # include all tokens if it's the prompt
        if not self.skip_prompt and self.next_tokens_are_prompt:
            self.token_buffer = value_as_list
            # keep a separate instance of the list, these needs to be unique objects
            self.all_buffer = list(value_as_list)
            self.next_tokens_are_prompt = False

        else:
            token = value_as_list[0]
            self.token_buffer.append(token)
            self.all_buffer.append(token)

        # detect if a space should be included, we use the previous value and if there's a space after it's length, then the current token gets text prepended

        text: str = self.tokenizer.decode(self.token_buffer, **self.decode_kwargs)

        if MODEL_LOGGING:
            print(text)

        # we hit a nonsense piece of text, so we wait for the next token
        if "�" in text:
            return

        formatted_text = text

        # this is to handle tokenizers that add spaces when there are two tokens that should map as having
        # a space between them
        if not self.next_tokens_are_prompt and not formatted_text.startswith(" "):

            prev_text = self.tokenizer.decode(
                self.all_buffer[:-1], **self.decode_kwargs
            )

            new_text = self.tokenizer.decode(self.all_buffer, **self.decode_kwargs)

            prev_text_len = len(prev_text)

            first_char_of_new_text = new_text[prev_text_len : prev_text_len + 1]

            if first_char_of_new_text == " ":
                formatted_text = f" {text}"

        self.text_queue.put(formatted_text)

        self.token_buffer = []

    def end(self):
        """If the stream is ending, also put a stop signal in the queue."""

        # flush the token list into text if tokens weren't cleared
        if self.token_buffer:
            text = self.tokenizer.decode(self.token_buffer, **self.decode_kwargs)

            self.text_queue.put(text)

        self.text_queue.put(self.stop_signal)

    def reset(self):
        self.text_queue = Queue()
        self.token_buffer = []
        self.next_tokens_are_prompt = True

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value
