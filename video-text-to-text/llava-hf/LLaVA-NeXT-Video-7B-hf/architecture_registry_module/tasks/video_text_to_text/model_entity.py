from typing import List
from dataclasses import dataclass
from io import BytesIO

import requests
import numpy as np

from architecture_registry_module.classes.model_entity import ModelEntity


@dataclass
class VideoTextToTextModelEntity(ModelEntity):
    def run_inference(self, *args, **kwargs):
        first_arg = args[0]

        if isinstance(first_arg, str):
            # this is the standard input, which is text and images
            return self.run_inference_default(*args, **kwargs)

        # supports chat messages
        first_arg: list
        return self.run_inference_chat(*args, **kwargs)

    def run_inference_default(self, text, videos=None, **kwargs):
        pass

    def run_inference_chat(self, messages, **kwargs):
        pass

    def adapt_to_conversational_chat_json(self, messages: List[dict]):
        new_messages = []
        videos = []

        for message in messages:

            new_content_items = []

            content = message["content"]

            if isinstance(content, str):
                new_content_items.append({"type": "text", "text": content})
            else:
                for content_item in message["content"]:
                    new_content_item = content_item

                    type = content_item["type"]

                    if type == "video":
                        video_url = content_item["url"]

                        new_content_item = {"type": "video"}

                        videos.append(video_url)

                    new_content_items.append(new_content_item)

            new_message = {**message, "content": new_content_items}

            new_messages.append(new_message)

        return new_messages, videos

    def download_video(self, video_url):
        response = requests.get(video_url)
        response.raise_for_status()
        video_bytes = BytesIO(response.content)

        return video_bytes

    def read_video_pyav(self, container, indices):
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


# universal stub used by the model loader
model_cls = VideoTextToTextModelEntity
