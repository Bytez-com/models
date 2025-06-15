from dataclasses import dataclass

from architecture_registry_module.tasks.image_text_to_text.model_entity import (
    ImageTextToTextModelEntity,
)


@dataclass
class Gemma3ForConditionalGeneration(ImageTextToTextModelEntity):
    def generate(self, text, images, videos, **kwargs):
        kwargs = {**{"generate_kwargs": {"streamer": kwargs.get("streamer")}, **kwargs}}

        if videos:
            kwargs["videos"] = videos

        # Save the original
        og_call = self.pipe.processor.__class__.__call__

        # Define the patch
        def patched_processor_call(self, *args, **kwargs):
            images = kwargs.get("images", None)
            if not images:
                kwargs["images"] = None
            return og_call(self, *args, **kwargs)

        # Apply it
        self.pipe.processor.__class__.__call__ = patched_processor_call

        output = self.pipe(text=text, images=images, **kwargs)

        generated_text = output[0]["generated_text"]

        return generated_text


# universal stub used by the model loader
model_cls = Gemma3ForConditionalGeneration
