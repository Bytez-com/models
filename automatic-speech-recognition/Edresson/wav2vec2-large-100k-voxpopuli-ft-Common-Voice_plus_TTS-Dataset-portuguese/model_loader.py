from transformers import pipeline
from environment import MODEL_ID, TASK, DEVICE

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
    "task": TASK,
    "model": MODEL_ID,
}


# attempt loading the model with the device_map
def try_device_map():
    def load_model_with_device_map(device):
        print("Attempting to load model via 'device_map' with device: ", device)
        return pipeline(**DEFAULT_KWARGS, device_map=device)

    pipe = _try_loading(pipeline_call=load_model_with_device_map)
    return pipe


# attempt loading the model with a specific device
def try_device():
    def load_model_with_device(device):
        print("Attempting to load model via 'device' with device: ", device)
        return pipeline(
            **DEFAULT_KWARGS,
            device=device,
        )

    pipe = _try_loading(pipeline_call=load_model_with_device)
    return pipe


# try the device_map first, then try the actual device
def try_loading(**kwargs):
    try:
        pipe = try_device_map(**kwargs)
        return pipe
    except Exception:
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
