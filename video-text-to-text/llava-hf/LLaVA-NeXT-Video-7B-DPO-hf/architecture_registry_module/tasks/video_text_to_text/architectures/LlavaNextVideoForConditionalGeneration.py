from dataclasses import dataclass

from architecture_registry_module.tasks.video_text_to_text.model_entity import (
    VideoTextToTextModelEntity,
)

from transformers import (
    LlavaNextVideoForConditionalGeneration as LlavaNextVideoForConditionalGenerationBaseClass,
)
import av
import numpy as np


@dataclass
class LlavaNextVideoForConditionalGeneration(VideoTextToTextModelEntity):
    @classmethod
    def load_model_from_model_id(cls, model_id: str, **kwargs):
        model = LlavaNextVideoForConditionalGenerationBaseClass.from_pretrained(
            model_id, **kwargs
        )
        return model

    pass

    def run_inference_default(self, text, videos, **kwargs):
        output = self.generate(text, videos, **kwargs)

        return output

    def run_inference_chat(self, *args, **kwargs):
        messages = args[0]

        adapted_messages, videos = self.adapt_to_conversational_chat_json(
            messages=messages
        )

        text = self.processor.apply_chat_template(
            adapted_messages, add_generation_prompt=True
        )

        output = self.generate(text, videos, **kwargs)

        output_messages = messages + [
            {"role": "assistant", "content": [{"type": "text", "text": output}]}
        ]

        return output_messages

    def generate(self, text, videos, **kwargs):
        if videos is None:
            videos = []

        kwargs = {**kwargs, "streamer": kwargs.get("streamer")}

        for index, video_url in enumerate(videos):
            video_bytes = self.download_video(video_url)

            container = av.open(video_bytes)

            # sample uniformly 8 frames from the video, can sample more for longer videos
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / 8).astype(int)
            clip = self.read_video_pyav(container, indices)

            videos[index] = clip


        if not videos:
            videos = None

        inputs_video = self.processor(
            text=text, videos=videos, padding=True, return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs_video, do_sample=False, **kwargs)

        new_tokens_len = len(output[0]) - len(inputs_video.input_ids[0])

        output_tokens = output[0][-new_tokens_len:]

        formatted_text = self.processor.decode(output_tokens, skip_special_tokens=True)

        return formatted_text


# universal stub used by the model loader
model_cls = LlavaNextVideoForConditionalGeneration
