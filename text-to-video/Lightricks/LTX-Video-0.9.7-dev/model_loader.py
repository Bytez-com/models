from typing import Any
import torch
import numpy as np
from dataclasses import dataclass
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video
from environment import MODEL_ID, DEVICE, MODEL_LOADING_KWARGS

_pipe = LTXConditionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="balanced", **MODEL_LOADING_KWARGS
)

pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
    "Lightricks/ltxv-spatial-upscaler-0.9.7",
    vae=_pipe.vae,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    **MODEL_LOADING_KWARGS,
)


device = "cuda" if DEVICE == "cuda" else "cpu"

_pipe.to(device)
pipe_upsample.to(device)
_pipe.vae.enable_tiling()


def round_to_nearest_resolution_acceptable_by_vae(height, width):
    height = height - (height % _pipe.vae_spatial_compression_ratio)
    width = width - (width % _pipe.vae_spatial_compression_ratio)
    return height, width


def pipe(prompt, **kwargs):
    downscale_factor = 2 / 3

    expected_height = kwargs.get("height", 704)
    expected_width = kwargs.get("height", 512)

    if kwargs.get("height"):
        del kwargs["height"]

    if kwargs.get("width"):
        del kwargs["width"]

    # Part 1. Generate video at smaller resolution
    downscaled_height = int(expected_height * downscale_factor)

    downscaled_width = int(expected_width * downscale_factor)

    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(
        downscaled_height, downscaled_width
    )

    latents = _pipe(
        prompt=prompt,
        width=downscaled_width,
        height=downscaled_height,
        generator=torch.Generator().manual_seed(0),
        output_type="latent",
        **kwargs,
    ).frames

    print("Done with latent")

    # Part 2. Upscale generated video using latent upsampler with fewer inference steps
    # The available latent upsampler upscales the height/width by 2x
    upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
    upscaled_latents = pipe_upsample(latents=latents, output_type="latent").frames

    if kwargs.get("num_inference_steps"):
        del kwargs["num_inference_steps"]

    # !!NOTE TODO!! AttributeError: 'list' object has no attribute 'frames' FIX THIS

    # Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
    frames = _pipe(
        prompt=prompt,
        width=upscaled_width,
        height=upscaled_height,
        denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
        num_inference_steps=10,
        latents=upscaled_latents,
        decode_timestep=0.05,
        image_cond_noise_scale=0.025,
        generator=torch.Generator().manual_seed(0),
        output_type="pil",
        **kwargs,
    ).frames[0]

    print("Done with denoising")

    # Resize each frame
    frames = [frame.resize((expected_width, expected_height)) for frame in frames]

    # Convert list of PIL to tensor (N, C, H, W)
    tensor_frames = torch.stack(
        [torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in frames]
    )

    # Safeguard for single-frame videos
    if tensor_frames.ndim == 3:
        tensor_frames = tensor_frames.unsqueeze(0)

    # Convert back to list of ndarrays (H, W, C) for export_to_video
    np_frames = [f.permute(1, 2, 0).cpu().numpy() for f in tensor_frames]

    video_result = VideoResult(np_frames=np_frames)

    return video_result


@dataclass
class VideoResult:
    np_frames: Any

    def squeeze(self):
        return self.np_frames

    @property
    def frames(self):
        return self


print("Model loaded")


def load_model():
    # instructions: do not modify this function
    # export the model loaded into global memory
    global pipe

    return pipe


if __name__ == "__main__":
    load_model()
