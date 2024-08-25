from transformers import pipeline
from environment import MODEL_ID, TASK, DEVICE


print("Loading model...")

# NOTE many text classification models do not support device_map="auto"
try:
    pipe = pipeline(
        TASK,
        model=MODEL_ID,
        ### params ###
        device_map="auto",
    )
except:  # noqa: E722
    pipe = pipeline(
        TASK,
        model=MODEL_ID,
        ### params ###
        device_map=DEVICE,
    )

print("Model loaded")


def load_model():
    # instructions: do not modify this function
    # export the model loaded into global memory
    global pipe

    return pipe


if __name__ == "__main__":
    load_model()
