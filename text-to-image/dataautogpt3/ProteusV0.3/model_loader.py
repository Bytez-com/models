from diffusers import DiffusionPipeline
from environment import MODEL_ID, DEVICE


print("Loading model...")

# NOTE diffusers only supports device_map="balanced", otherwise you have to put weights explicitly on the cpu or gpu
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    ### params ###
    # device_map="balanced",
).to(DEVICE)


print("Model loaded")


def load_model():
    # instructions: do not modify this function
    # export the model loaded into global memory
    global pipe

    return pipe


if __name__ == "__main__":
    load_model()
