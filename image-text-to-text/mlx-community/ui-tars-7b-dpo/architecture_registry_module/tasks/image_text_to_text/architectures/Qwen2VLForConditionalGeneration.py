from dataclasses import dataclass

from PIL import Image
import requests

from architecture_registry_module.tasks.image_text_to_text.model_entity import (
    ImageTextToTextModelEntity,
)

from transformers import (
    Qwen2VLForConditionalGeneration as _Qwen2VLForConditionalGeneration,
)

from torch import nn


class MonkeyPatchedEmbedding(nn.Module):
    def __init__(self, original_embed_tokens, device):
        super().__init__()
        self.original_embed_tokens = original_embed_tokens
        self.device = device

    def forward(self, *args, **kwargs):
        # Use the original embedding layer but ensure output is on the correct device
        tensors = self.original_embed_tokens(*args, **kwargs)
        return tensors.to(self.device)


@dataclass
class Qwen2VLForConditionalGeneration(ImageTextToTextModelEntity):
    @classmethod
    def load_model_from_model_id(cls, model_id: str, **kwargs):
        model = _Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        return model

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

        model = cls.load_model_from_model_id(model_id, **kwargs)

        # NOTE allows multi GPU to work properyl, monkey patch the bad embed layer that points to device 1, the forward() logic demands that
        # these be on the same device as operations are performed on the attention mask and input_ids, which reside on the model's default device
        # the problem is accelerate gets that one aspect of the device map wrong when it tries to infer where each layer should go.
        # we force that layer to be on the same layer as the rest of the inputs with this monkey patch
        original_embed_tokens = model.model.embed_tokens

        model.embed_tokens = MonkeyPatchedEmbedding(original_embed_tokens, model.device)

        return cls(model=model, processor=processor)

    def generate(self, text, images, videos, **kwargs):
        text_prompt = self.processor.apply_chat_template(
            text, add_generation_prompt=True
        )

        images = [Image.open(requests.get(url, stream=True).raw) for url in images]

        if not images:
            images = None

        inputs = self.processor(
            text=[text_prompt], images=images, padding=True, return_tensors="pt"
        )

        inputs = inputs.to(self.model.device)

        # Generate outputs
        generated_ids = self.model.generate(
            **inputs,
            **kwargs,
        )

        input_ids = inputs.input_ids[0]

        new_generated_ids = generated_ids[0][len(input_ids) :]

        formatted_text: str = self.processor.decode(
            new_generated_ids, skip_special_tokens=True
        )

        formatted_text = formatted_text.strip()

        # if the inference was submitted as chat
        if isinstance(text, list):
            return [{"role": "assistant", "content": formatted_text}]

        return formatted_text


# universal stub used by the model loader
model_cls = Qwen2VLForConditionalGeneration
