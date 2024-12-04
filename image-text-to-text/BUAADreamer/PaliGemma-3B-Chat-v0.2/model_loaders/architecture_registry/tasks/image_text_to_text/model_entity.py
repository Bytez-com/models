from dataclasses import dataclass
from transformers import AutoModelForVision2Seq

from model_loaders.architecture_registry.classes.model_entity import ModelEntity


@dataclass
class TextToTextToVisionModelEntity(ModelEntity):

    @classmethod
    def load_model_from_model_id(cls, model_id: str, **kwargs):
        model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
        return model
