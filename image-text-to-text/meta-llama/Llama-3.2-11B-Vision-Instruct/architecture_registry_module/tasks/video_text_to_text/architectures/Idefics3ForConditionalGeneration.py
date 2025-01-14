from dataclasses import dataclass
from typing import List

from architecture_registry_module.tasks.video_text_to_text.model_entity import (
    VideoTextToTextModelEntity,
)

from transformers import AutoModelForVision2Seq

import av
import numpy as np
from PIL import Image


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


@dataclass
class Idefics3ForConditionalGeneration(VideoTextToTextModelEntity):
    @classmethod
    def load_model_from_model_id(cls, model_id: str, **kwargs):
        model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
        return model

    def run_inference_default(self, text, videos, **kwargs):
        return self.generate(text, videos, **kwargs)

    def run_inference_chat(self, *args, **kwargs):
        messages = args[0]

        adapted_messages, videos = self.adapt_to_conversational_chat_json(
            messages=messages
        )

        # Prepare inputs
        prompt = self.processor.apply_chat_template(
            adapted_messages, add_generation_prompt=True
        )

        output = self.generate(prompt, videos, **kwargs)

        return output

    def generate(self, text, videos, **kwargs):

        for index, video_url in enumerate(videos):
            video_bytes = self.download_video(video_url)

            container = av.open(video_bytes)

            # sample uniformly 8 frames from the video, can sample more for longer videos
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / 8).astype(int)
            clip = read_video_pyav(container, indices)

            videos[index] = clip

        videos = [Image.fromarray(video[0]) for video in videos]

        inputs = self.processor(text=text, images=videos, return_tensors="pt")
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
            new_generated_ids, skip_special_tokens=False
        )

        formatted_text = formatted_text.strip()

        return formatted_text

    def adapt_to_conversational_chat_json(self, messages: List[dict]):
        adapted_messages = []
        videos = []

        text_content = ""

        for message in messages:
            image_content_items = []
            new_content_items = []

            for content_item in message["content"]:
                type = content_item["type"]
                

                if type == "text":
                    content = content_item[type]
                    new_content_items.append({"type": "text", "text": content})
                    text_content += content

                # NOTE these models are special, they still use the "image" token
                elif type == "video":
                    content = content_item['url']
                    image_content_items.append({"type": "image"})
                    videos.append(content)

            adapted_messages.append(
                {
                    "role": message["role"],
                    # NOTE videos must come before all other content items with these
                    "content": image_content_items + new_content_items,
                }
            )

        return adapted_messages, videos


# universal stub used by the model loader
model_cls = Idefics3ForConditionalGeneration
