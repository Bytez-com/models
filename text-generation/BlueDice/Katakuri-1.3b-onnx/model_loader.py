from optimum.pipelines import pipeline
from environment import MODEL_ID, TASK

print("Loading model...")

pipe = pipeline(
    TASK,
    model=MODEL_ID,
    accelerator="ort",
  ### params ###
    device_map="auto",
)

print("Model loaded")



if TASK in ["text-generation"]:
    from streamer import SingleTokenStreamer
    
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
