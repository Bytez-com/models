from dataclasses import dataclass

from architecture_registry_module.tasks.image_text_to_text.model_entity import (
    ImageTextToTextModelEntity,
)


@dataclass
class PaliGemmaModelEntity(ImageTextToTextModelEntity):
    def run_inference_default(self, text, images, videos=None, **kwargs):
        output = self.generate(text, images, videos, **kwargs)

        formatted_output = output[len(text):]

        return formatted_output
    


# universal stub used by the model loader
model_cls = PaliGemmaModelEntity
