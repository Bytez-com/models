from dataclasses import dataclass


from model_loaders.architecture_registry.tasks.image_text_to_text.sub_tasks.chat.model_entity import (
    TextToTextToVisionChatModelEntity,
)


@dataclass
class PaliGemmaModelEntity(TextToTextToVisionChatModelEntity):
    pass


model_cls = PaliGemmaModelEntity
