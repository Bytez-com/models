from dataclasses import dataclass

from architecture_registry_module.tasks.audio_text_to_text.model_entity import (
    AudioTextToTextModelEntity,
)

from transformers import (
    Qwen2AudioForConditionalGeneration as Qwen2AudioForConditionalGenerationBaseClass,
)


@dataclass
class Qwen2AudioForConditionalGeneration(AudioTextToTextModelEntity):
    @classmethod
    def load_model_from_model_id(cls, model_id: str, **kwargs):
        model = Qwen2AudioForConditionalGenerationBaseClass.from_pretrained(
            model_id, **kwargs
        )
        return model

    pass


# universal stub used by the model loader
model_cls = Qwen2AudioForConditionalGeneration
