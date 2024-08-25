from sentence_transformers import SentenceTransformer
from environment import MODEL_ID, DEVICE

print("Loading model...")

# construct as a set to dedupe, then turn into list
DEVICES = list(
    {
        "auto",
        # if the device was specified, i.e. cuda for instances, then if auto fails, this will run
        DEVICE,
        # always fallback to cpu worst case scenario
        "cpu",
    }
)


# wrapper for loading model on a specific device
def _try_loading(pipeline_call: callable):
    collected_exception = None

    for device in DEVICES:
        try:
            pipe = pipeline_call(device)

            print("Loaded model on device: ", device)

            return pipe
        except Exception as exception:
            collected_exception = exception

    raise collected_exception


DEFAULT_KWARGS = {
    ### params ###
    "model_name_or_path": MODEL_ID,
}


# attempt loading the model with a specific device
def try_device():
    def load_model_with_device(device):
        print("Attempting to load model via 'device' with device: ", device)
        return SentenceTransformer(
            **DEFAULT_KWARGS,
            device=device,
        )

    pipe = _try_loading(pipeline_call=load_model_with_device)
    return pipe


# NOTE sentence similarity models do not support device_map, only device
def try_loading(**kwargs):
    pipe = try_device(**kwargs)
    return pipe


pipe = try_loading()


print("Model loaded")


def load_model():
    # instructions: do not modify this function
    # export the model loaded into global memory
    global pipe

    return pipe


if __name__ == "__main__":
    load_model()
