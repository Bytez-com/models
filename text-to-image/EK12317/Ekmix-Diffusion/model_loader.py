from diffusers import DiffusionPipeline
from environment import MODEL_ID, DEVICE
from transformers import CLIPTokenizer, CLIPImageProcessor


print("Loading model...")

DEFAULT_KWARGS = {
    ### params ###
    "device_map": "balanced",
    "tokenizer": CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
}

try:
    # NOTE diffusers only supports device_map="balanced", .to() method can be used to specify a specific device
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, **DEFAULT_KWARGS)
except Exception:
    # NOTE if loading with the device map failed, try loading on a specific device
    del DEFAULT_KWARGS["device_map"]
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, **DEFAULT_KWARGS).to(DEVICE)


print("Model loaded")


def load_model():
    # instructions: do not modify this function
    # export the model loaded into global memory
    global pipe

    return pipe


if __name__ == "__main__":
    load_model()
