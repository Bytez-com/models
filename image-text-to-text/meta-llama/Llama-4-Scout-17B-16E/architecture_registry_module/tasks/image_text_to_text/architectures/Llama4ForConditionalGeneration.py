from dataclasses import dataclass

from architecture_registry_module.tasks.image_text_to_text.model_entity import (
    ImageTextToTextModelEntity,
)


@dataclass
class Llama4ModelEntity(ImageTextToTextModelEntity):
    def run_inference_default(self, text, images, videos=None, **kwargs):
        output = self.generate(text, images, videos, **kwargs)

        return output

    def run_inference_chat(self, *args, **kwargs):
        messages = args[0]

        adapted_messages, images, videos = self.adapt_to_conversational_chat_json(
            messages=messages
        )

        output = self.generate(adapted_messages, images, videos, **kwargs)

        last_message = output[-1]

        last_message_content = last_message["content"]

        last_message["role"] = "assistant"
        last_message["content"] = [{"type": "text", "text": last_message_content}]

        return output
    


# universal stub used by the model loader
model_cls = Llama4ModelEntity
