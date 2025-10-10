from dataclasses import dataclass

from architecture_registry_module.tasks.image_text_to_text.model_entity import (
    ImageTextToTextModelEntity,
)


@dataclass
class MllamaModelEntity(ImageTextToTextModelEntity):
    def run_inference_default(self, text, images, videos=None, **kwargs):
        kwargs = {**{"generate_kwargs": {"streamer": kwargs.get("streamer")}, **kwargs}}

        image_token = "<|image|>"

        if isinstance(images, str):
            image_tokens = [image_token]
        else:
            image_tokens = map(lambda _: image_token, images)

        image_tokens = "".join(image_tokens)

        text = f"{image_tokens}<|begin_of_text|>{text}"
        output = self.pipe(text=text, images=images, videos=videos, **kwargs)

        return output


# universal stub used by the model loader
model_cls = MllamaModelEntity
