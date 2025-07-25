import os
from typing import Any
from dataclasses import dataclass
import torch
from diffusers import LTXConditionPipeline, AutoModel
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.hooks import apply_group_offloading
from environment import MODEL_ID, MODEL_LOADING_KWARGS
from diffusers.utils import export_to_video, load_image, load_video


WORKING_DIR = os.path.dirname(os.path.realpath(__file__))

# Load transformer with FP8 casting
transformer = AutoModel.from_pretrained(
    MODEL_ID,
    **dict(subfolder="transformer", torch_dtype=torch.bfloat16, **MODEL_LOADING_KWARGS),
)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
)

# Load pipeline
pipeline = LTXConditionPipeline.from_pretrained(
    MODEL_ID,
    **dict(transformer=transformer, torch_dtype=torch.bfloat16, **MODEL_LOADING_KWARGS),
)

# Apply group-offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

pipeline.transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True,
)
apply_group_offloading(
    pipeline.text_encoder,
    onload_device=onload_device,
    offload_type="block_level",
    num_blocks_per_group=2,
)
apply_group_offloading(
    pipeline.vae, onload_device=onload_device, offload_type="leaf_level"
)


@dataclass
class VideoResult:
    np_frames: Any

    def squeeze(self):
        return self.np_frames

    @property
    def frames(self):
        return self


def pipe(prompt: str, **kwargs):
    conditions = None

    if "image_url" in kwargs:
        image_url = kwargs["image_url"]

        image = load_image(image_url)

        video = load_video(
            export_to_video([image])
        )  # compress the image using video compression as the model was trained on videos
        condition1 = LTXVideoCondition(video=video, frame_index=0)

        conditions = [condition1]

        del kwargs["image_url"]

    generator = None

    if "seed" in kwargs:
        seed = kwargs["seed"]
        generator = torch.Generator().manual_seed(seed)

        del kwargs["seed"]

    updated_kwargs = {
        **dict(
            conditions=conditions,
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
            num_frames=60,
            num_inference_steps=50,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            width=704,
            height=480,
            guidance_scale=3,
            generator=generator,
        )
    }

    for key, value in kwargs.items():
        updated_kwargs[key] = value

    # Run pipeline
    output = pipeline(**updated_kwargs)

    video = output.frames[0]

    export_to_video(video, f"{WORKING_DIR}/output.mp4", fps=24)

    # Return wrapped result
    video_result = VideoResult(np_frames=video)
    return video_result


print("Model loaded")


def load_model():
    # instructions: do not modify this function
    # export the model loaded into global memory
    global pipe
    return pipe


if __name__ == "__main__":
    load_model()
