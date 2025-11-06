from dataclasses import dataclass

from architecture_registry_module.tasks.image_text_to_text.model_entity import (
    ImageTextToTextModelEntity,
)


@dataclass
class PaliGemmaModelEntityForConditionalGeneration(ImageTextToTextModelEntity):
    def generate(self, text, images, videos, **kwargs):
        kwargs = {**{"generate_kwargs": {"streamer": kwargs.get("streamer")}, **kwargs}}

        if videos:
            kwargs["videos"] = videos

        # Save the original
        og_call = self.pipe.processor.__class__.__call__

        # Define the patch, this allows us to pass in None if there are no images
        def patched_processor_call(self, *args, **kwargs):
            images = kwargs.get("images", None)
            if not images:
                kwargs["images"] = None

            results = og_call(self, *args, **kwargs)

            return results

        # Apply it
        self.pipe.processor.__class__.__call__ = patched_processor_call

        images = self.load_images(images)

        output = self.pipe(text=text, images=images, **kwargs)

        self.cleanup_output(output)

        return output

    # for whatever reason this model returns the actual PIL image instead of the original text
    # because the OAI standard and our standard is only concerned with returning the last message
    # this should have no effect outside of maybe influencing the total token count
    def cleanup_output(self, output):
        for completion in output:
            for message in completion["input_text"]:
                self._cleanup_output(message)
            for message in completion["generated_text"]:
                self._cleanup_output(message)

    def _cleanup_output(self, message):
        for item in message["content"]:
            if item["type"] == "image":
                item["image"] = "IMG"


# universal stub used by the model loader
model_cls = PaliGemmaModelEntityForConditionalGeneration
