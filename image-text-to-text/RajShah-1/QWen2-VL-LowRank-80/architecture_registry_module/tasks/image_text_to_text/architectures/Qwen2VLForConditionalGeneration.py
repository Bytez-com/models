from dataclasses import dataclass

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
        # we only update it if it exists on the model

        if getattr(model.model, "embed_tokens", None):
            original_embed_tokens = model.model.embed_tokens

            model.embed_tokens = MonkeyPatchedEmbedding(
                original_embed_tokens, model.device
            )

        return cls(model=model, processor=processor)

    def run_inference_default(self, text, images, **kwargs):
        return self.generate(text, images, **kwargs)

    def run_inference_chat(self, *args, **kwargs):
        messages = args[0]

        adapted_messages, images, videos = self.adapt_to_conversational_chat_json(
            messages=messages
        )

        # output is a dict that contains keys "generated_text", "scores", "sequence" etc. if called directly from transformers pipeline() pipe
        output = self.generate(adapted_messages, images, **kwargs)[0]

        output_messages = messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": output["generated_text"]}],
            }
        ]

        return [{**output, "generated_text": output_messages}]

    def generate(self, text, images, **kwargs):
        text_prompt = self.processor.apply_chat_template(
            text, add_generation_prompt=True
        )

        loaded_images = self.load_images(images)

        if not loaded_images:
            loaded_images = None

        inputs = self.processor(
            text=[text_prompt],
            images=loaded_images,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        # Generate outputs
        generated_ids = self.model.generate(
            **inputs,
            **kwargs,
        )

        input_ids = inputs.input_ids[0]

        new_generated_ids = generated_ids["sequences"][0][len(input_ids) :]

        formatted_text: str = self.processor.decode(
            new_generated_ids, skip_special_tokens=True
        )

        formatted_text = formatted_text.strip()

        return [
            dict(
                generated_text=formatted_text,
                sequence=generated_ids.get("sequences", [[]])[0],
                scores=generated_ids.get("scores", None),
            )
        ]


# universal stub used by the model loader
model_cls = Qwen2VLForConditionalGeneration
