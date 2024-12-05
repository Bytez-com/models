from typing import List, Dict
from dataclasses import dataclass
from transformers import AutoModelForVision2Seq

from architecture_registry_module.classes.model_entity import ModelEntity
from architecture_registry_module.classes.messages import Messages


from PIL import Image

import requests


@dataclass
class ImageTextToTextModelEntity(ModelEntity):
    image_token: str = "<|image|><|begin_of_text|>"

    @classmethod
    def load_model_from_model_id(cls, model_id: str, **kwargs):
        model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
        return model

    def __call__(self, *args, **kwargs):
        return self.run_inference(*args, **kwargs)

    def run_inference(self, *args, **kwargs):
        first_arg = args[0]

        if isinstance(first_arg, str):
            # this is the standard input, which is text and images
            return self.run_inference_default(*args, **kwargs)

        # supports chat messages
        return self.run_inference_chat(*args, **kwargs)

    def run_inference_default(self, text, images, **kwargs):

        text_with_token = f"{self.image_token}{text}"

        if isinstance(images, str):
            image = Image.open(requests.get(images, stream=True).raw)
            images = [image]

        else:
            new_images = []

            for image in images:
                pil_image = Image.open(requests.get(images, stream=True).raw)
                new_images.append(pil_image)

            images = new_images

        inputs = self.processor(
            text=text_with_token, images=images, return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            **kwargs,
        )

        decoded_ouput: str = self.processor.decode(
            output[0],
            #  strips formatting tokens used by the model to understand its inputs
            skip_special_tokens=True,
        )

        # NOTE it's unclear as to whether or not the chat template applies white space, but for the sake of the user's sanity
        # and our own, we strip the ends
        sliced_output = decoded_ouput[len(text) :].strip()

        return sliced_output

    def run_inference_chat(self, *args, **kwargs):
        messages = args[0]

        input_str_no_special_tokens, input_str, images = (
            self.pre_process_conversational(messages=messages)
        )

        inputs = self.processor(text=input_str, images=images, return_tensors="pt").to(
            self.model.device
        )

        output = self.model.generate(
            **inputs,
            **kwargs,
        )

        decoded_ouput: str = self.processor.decode(
            output[0],
            #  strips formatting tokens used by the model to understand its inputs
            skip_special_tokens=True,
        )

        # NOTE it's unclear as to whether or not the chat template applies white space, but for the sake of the user's sanity
        # and our own, we strip the ends
        sliced_output = decoded_ouput[len(input_str_no_special_tokens) :].strip()

        return sliced_output

    def pre_process_conversational(self, messages: List[Dict]):

        messages = Messages.from_json_list(messages=messages)

        adapted_messages, images = self.adapt_to_conversational_chat_json(
            messages=messages
        )

        input_ids = self.processor.tokenizer.apply_chat_template(
            adapted_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        input_str_no_special_tokens = self.processor.decode(
            input_ids[0],
            #  strips formatting tokens used by the model to understand its inputs
            skip_special_tokens=True,
        )

        input_str = self.processor.decode(
            input_ids[0],
            #  strips formatting tokens used by the model to understand its inputs
            # skip_special_tokens=True,
        )

        return input_str_no_special_tokens, input_str, images

    def adapt_to_conversational_chat_json(self, messages: Messages):
        adapted_messages = []
        images = []

        for message in messages.items:
            # accumulate the message as strings
            message_strings = []

            for content_item in message.content_items:
                type = content_item.type
                content = content_item.value

                if type == "text":
                    message_strings.append(content)

                elif type == "image_url":
                    image = Image.open(requests.get(content, stream=True).raw)
                    images.append(image)

        message_text = "".join(message_strings)

        adapted_messages.append({"role": message.role, "content": message_text})

        return adapted_messages, images
