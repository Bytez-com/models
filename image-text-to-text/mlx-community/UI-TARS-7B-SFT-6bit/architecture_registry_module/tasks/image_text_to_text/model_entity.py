import base64
from typing import List
from dataclasses import dataclass
from io import BytesIO


from PIL import Image
import requests
from transformers import AutoProcessor, pipeline

from architecture_registry_module.classes.model_entity import ModelEntity


@dataclass
class ImageTextToTextModelEntity(ModelEntity):
    @classmethod
    def load_processor_from_model_id(cls, model_id: str, **kwargs):
        # limit the input image size to 256 min pixels and 1080 max pixels to prevent gpu OOM
        min_pixels = 256 * 28 * 28
        max_pixels = 1080 * 28 * 28

        processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs
        )

        return processor

    @classmethod
    def load_from_model_id(
        cls, model_id: str, load_model=True, load_processor=True, **kwargs
    ):

        processor = (
            cls.load_processor_from_model_id(model_id, **kwargs)
            if load_processor
            else None
        )

        # some models do not include a chat_template (they're missing "chat_template.json"), we default to the tokenizer's
        processor.chat_template = (
            processor.chat_template or processor.tokenizer.chat_template
        )

        if not load_model:
            return cls(model=None, processor=processor, pipe=None, **kwargs)

        pipe = pipeline(
            "image-text-to-text", model=model_id, processor=processor, **kwargs
        )

        return cls(model=pipe.model, processor=processor, pipe=pipe)

    def __call__(self, *args, **kwargs):
        return self.run_inference(*args, **kwargs)

    def run_inference(self, *args, **kwargs):
        first_arg = args[0]

        if isinstance(first_arg, str):
            # this is the standard input, which is text and images
            return self.run_inference_default(*args, **kwargs)

        # supports chat messages
        first_arg: list
        return self.run_inference_chat(*args, **kwargs)

    def run_inference_default(self, text, images, videos=None, **kwargs):
        output = self.generate(text, images, videos, **kwargs)

        return output

    def run_inference_chat(self, *args, **kwargs):
        messages = args[0]

        adapted_messages, images, videos = self.adapt_to_conversational_chat_json(
            messages=messages
        )

        # output is a dict that contains keys "generated_text", "scores", "sequence" etc. if called directly from transformers pipeline() pipe
        output = self.generate(adapted_messages, images, videos, **kwargs)[0]

        return [output]

    def generate(self, text, images, videos, **kwargs):
        kwargs = {**{"generate_kwargs": {"streamer": kwargs.get("streamer")}, **kwargs}}

        if videos:
            kwargs["videos"] = videos

        loaded_images = self.load_images(images)

        # NOTE if images is an empty array, set it to none, it will fail on some models otherwise
        if not loaded_images:
            loaded_images = None

        output = self.pipe(text=text, images=loaded_images, **kwargs)

        return output

    def load_images(self, images):
        loaded_images = []
        for image_string in images:
            image_string: str
            # normal http links
            if image_string.startswith("http"):
                image = Image.open(requests.get(image_string, stream=True).raw)
                loaded_images.append(image)
                continue

            # base64 strings
            if "," in image_string:
                b64_string = image_string.split(",", 1)[1]
            else:
                b64_string = image_string
            image_data = base64.b64decode(b64_string)
            image = Image.open(BytesIO(image_data))
            loaded_images.append(image)

        return loaded_images

    def adapt_to_conversational_chat_json(self, messages: List[dict]):
        new_messages = []
        images = []
        videos = []

        for message in messages:

            new_content_items = []

            content = message["content"]

            if isinstance(content, str):
                new_content_items.append({"type": "text", "text": content})

            else:
                for content_item in message["content"]:
                    new_content_item = content_item

                    type = content_item["type"]

                    if type == "image":
                        image_url = content_item.get("url") or content_item.get(
                            "base64"
                        )

                        images.append(image_url)
                        new_content_item = {"type": "image"}

                    # some models can take in videos as multiple images
                    if type == "video":
                        video_url = content_item.get("url") or content_item.get(
                            "base64"
                        )
                        new_content_item = {"type": "image"}

                        videos.append(video_url)

                    new_content_items.append(new_content_item)

            new_message = {**message, "content": new_content_items}

            new_messages.append(new_message)

        return new_messages, images, videos


# universal stub used by the model loader
model_cls = ImageTextToTextModelEntity
