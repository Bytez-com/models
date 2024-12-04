from dataclasses import dataclass


from model_loaders.architecture_registry.tasks.image_text_to_text.sub_tasks.default.model_entity import (
    TextToTextToVisionDefaultModelEntity,
)


@dataclass
class PaliGemmaModelEntity(TextToTextToVisionDefaultModelEntity):
    pass


model_cls = PaliGemmaModelEntity
