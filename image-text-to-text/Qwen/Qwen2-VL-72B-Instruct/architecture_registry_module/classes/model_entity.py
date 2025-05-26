from typing import Callable
from dataclasses import dataclass, field
from transformers import PreTrainedModel, AutoProcessor, AutoModel


@dataclass
class ModelEntity:
    model: PreTrainedModel
    processor: Callable
    pipe: Callable = None
    _forward_params: dict = field(init=True, default_factory=dict)

    @classmethod
    def load_model_from_model_id(cls, model_id: str, **kwargs):
        model = AutoModel.from_pretrained(model_id, **kwargs)
        return model

    @classmethod
    def load_processor_from_model_id(cls, model_id: str, **kwargs):
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        return processor

    @classmethod
    def load_from_model_id(
        cls, model_id: str, load_model=True, load_processor=True, **kwargs
    ):
        model = cls.load_model_from_model_id(model_id, **kwargs) if load_model else None
        processor = (
            cls.load_processor_from_model_id(model_id, **kwargs)
            if load_processor
            else None
        )

        return cls(model=model, processor=processor)

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def run_inference(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        kwargs = {**kwargs, **self._forward_params}

        if self.pipe:
            return self.pipe(*args, **kwargs)

        return self.run_inference(*args, **kwargs)
