from dataclasses import dataclass

from architecture_registry_module.tasks.image_text_to_text.model_entity import (
    ImageTextToTextModelEntity,
)


@dataclass
class Llama4ModelEntity(ImageTextToTextModelEntity):
    pass


# universal stub used by the model loader
model_cls = Llama4ModelEntity
