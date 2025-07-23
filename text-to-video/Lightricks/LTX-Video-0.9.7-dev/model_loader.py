from typing import Any
from dataclasses import dataclass
import torch
from diffusers import LTXPipeline, AutoModel
from diffusers.hooks import apply_group_offloading
from environment import MODEL_ID, MODEL_LOADING_KWARGS


# Load transformer with FP8 casting
transformer = AutoModel.from_pretrained(
    MODEL_ID,
    **dict(subfolder="transformer", torch_dtype=torch.bfloat16, **MODEL_LOADING_KWARGS),
)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
)

# Load pipeline
pipeline = LTXPipeline.from_pretrained(
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
    updated_kwargs = {
        **dict(
            prompt=prompt,
            num_frames=24,
            num_inference_steps=30,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            width=768,
            height=512,
            **kwargs,
        )
    }

    # Run pipeline
    output = pipeline(**updated_kwargs)

    video = output.frames[0]

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
