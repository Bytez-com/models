from dataclasses import dataclass
from typing import List

from architecture_registry_module.tasks.image_text_to_text.model_entity import (
    ImageTextToTextModelEntity,
)


@dataclass
class Idefics3ForConditionalGeneration(ImageTextToTextModelEntity):
    def run_inference_default(self, text, images, **kwargs):
        return self.generate(text, images, **kwargs)

    def run_inference_chat(self, *args, **kwargs):
        messages = args[0]

        adapted_messages, images = self.adapt_to_conversational_chat_json(
            messages=messages
        )

        # Prepare inputs
        prompt = self.processor.apply_chat_template(
            adapted_messages, add_generation_prompt=True
        )

        output = self.generate(prompt, images, **kwargs)[0]

        output_messages = messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": output["generated_text"]}],
            }
        ]

        return [{**output, "generated_text": output_messages}]

    def generate(self, text, images, **kwargs):
        loaded_images = self.load_images(images)

        # This model does an explicit check for None in its proocessor, otherwise it errors
        if not loaded_images:
            loaded_images = None

        inputs = self.processor(text=text, images=loaded_images, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        # Generate outputs
        generated_ids = self.model.generate(
            **inputs,
            **kwargs,
        )

        # this returns only the new ids that were generated for the sequence of inputs
        input_ids = inputs.input_ids[0]

        new_generated_ids = generated_ids[0][len(input_ids) :]

        formatted_text: str = self.processor.decode(
            new_generated_ids, skip_special_tokens=True
        )

        formatted_text = formatted_text.strip()

        return [
            dict(
                generated_text=formatted_text,
                sequence=getattr(generated_ids, "sequences", [[]])[0],
                scores=getattr(generated_ids, "scores", None),
            )
        ]

    def adapt_to_conversational_chat_json(self, messages: List[dict]):
        adapted_messages = []
        images = []

        text_content = ""

        for message in messages:
            image_content_items = []
            new_content_items = []

            content = message["content"]

            if isinstance(content, str):
                new_content_items.append({"type": "text", "text": content})

            else:
                for content_item in message["content"]:
                    type = content_item["type"]

                    if type == "text":
                        content = content_item["text"]
                        new_content_items.append({"type": "text", "text": content})
                        text_content += content

                    elif type == "image":
                        image_url = content_item.get("url") or content_item.get(
                            "base64"
                        )
                        image_content_items.append({"type": "image"})
                        images.append(image_url)

            adapted_messages.append(
                {
                    "role": message["role"],
                    # NOTE images must come before all other content items with these
                    "content": image_content_items + new_content_items,
                }
            )

        return adapted_messages, images


# universal stub used by the model loader
model_cls = Idefics3ForConditionalGeneration
