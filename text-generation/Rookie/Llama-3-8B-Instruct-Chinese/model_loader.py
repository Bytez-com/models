from os import environ
from transformers import pipeline, AutoTokenizer
from streamer import SingleTokenStreamer

MODEL_ID = environ.get("MODEL_ID", "")
TASK = environ.get("TASK", "")

stream_supported = ['text-generation']

if TASK in stream_supported:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_ID)
    streamer = SingleTokenStreamer(
        tokenizer=tokenizer, skip_prompt=False, skip_special_tokens=True
    )


pipe = pipeline(
    TASK,
    model=MODEL_ID,
    ### params ###
    device_map="auto",
    trust_remote_code=True,
    streamer=streamer if TASK in stream_supported else None,
)


def load_model():
    # instructions: do not modify this function
    # export the model loaded into global memory
    global pipe

    return pipe


if __name__ == "__main__":
    load_model()
