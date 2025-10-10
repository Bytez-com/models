from dataclasses import dataclass

from PIL import Image
import requests

from architecture_registry_module.tasks.image_text_to_text.model_entity import (
    ImageTextToTextModelEntity,
)

from transformers import (
    Gemma3ForConditionalGeneration as _Gemma3ForConditionalGeneration,
)


@dataclass
class Gemma3ForConditionalGeneration(ImageTextToTextModelEntity):
    @classmethod
    def load_model_from_model_id(cls, model_id: str, **kwargs):
        model = _Gemma3ForConditionalGeneration.from_pretrained(model_id, **kwargs)
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

        return cls(model=model, processor=processor)

    def run_inference_chat(self, *args, **kwargs):
        messages = args[0]

        adapted_messages, images, videos = self.adapt_to_conversational_chat_json(
            messages=messages
        )

        # output is a dict that contains keys "generated_text", "scores", "sequence" etc. if called directly from transformers pipeline() pipe
        output = self.generate(adapted_messages, images, videos, **kwargs)[0]

        output_messages = messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": output["generated_text"]}],
            }
        ]

        return [{**output, "generated_text": output_messages}]

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
model_cls = Gemma3ForConditionalGeneration
