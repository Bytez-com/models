from transformers import pipeline
from environment import MODEL_ID, TASK
from streamer import SingleTokenStreamer


print("Loading model...")

pipe = pipeline(
    TASK,
    model=MODEL_ID,
    ### params ###
  device_map = "auto",
  trust_remote_code = True
)

print("Model loaded")

if TASK in ["text-generation"]:
    streamer = SingleTokenStreamer(
        tokenizer=pipe.tokenizer, skip_prompt=False, skip_special_tokens=True
    )

    pipe._forward_params = {**pipe._forward_params, "streamer": streamer}


def load_model():
    # instructions: do not modify this function
    # export the model loaded into global memory
    global pipe

    return pipe


if __name__ == "__main__":
    load_model()
