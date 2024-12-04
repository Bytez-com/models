from dataclasses import dataclass


from model_loaders.architecture_registry.tasks.image_text_to_text.model_entity import (
    TextToTextToVisionModelEntity,
)

from PIL import Image

import requests


@dataclass
class TextToTextToVisionDefaultModelEntity(TextToTextToVisionModelEntity):
    image_token: str = "<|image|><|begin_of_text|>"

    def __call__(self, *args, **kwargs):
        return self.run_inference(*args, **kwargs)

    def run_inference(
        self: "TextToTextToVisionDefaultModelEntity", text, images, **kwargs
    ):

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
