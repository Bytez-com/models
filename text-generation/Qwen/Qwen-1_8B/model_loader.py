from collections import OrderedDict
from transformers import pipeline
from environment import MODEL_ID, TASK, DEVICE


print("Loading model...")

# construct as a set to dedupe, then turn into list
initial_devices = [
    "auto",
    # if the device was specified, i.e. cuda for instances, then if auto fails, this will run
    DEVICE,
    # always fallback to cpu worst case scenario
    "cpu",
]

# Use OrderedDict to maintain order while deduplicating
DEVICES = list(OrderedDict.fromkeys(initial_devices))


# wrapper for loading model on a specific device
def _try_loading(pipeline_call: callable, device_loader: str):
    collected_exception = None

    for device in DEVICES:
        try:
            print(
                f"Attempting to load model via '{device_loader}' with device: ", device
            )
            pipe = pipeline_call(device)

            print(f"Loaded model via '{device_loader}' on device: ", device)

            return pipe
        except Exception as exception:
            collected_exception = exception

    raise collected_exception


DEFAULT_KWARGS = {
    ### params ###
    "task": TASK,
    "model": MODEL_ID,
    "trust_remote_code": True
}


# attempt loading the model with the device_map
def try_device_map():
    def load_model_with_device_map(device):
        return pipeline(**DEFAULT_KWARGS, device_map=device)

    pipe = _try_loading(
        pipeline_call=load_model_with_device_map, device_loader="device_map"
    )
    return pipe


# attempt loading the model with a specific device
def try_device():
    def load_model_with_device(device):
        return pipeline(
            **DEFAULT_KWARGS,
            device=device,
        )

    pipe = _try_loading(
        pipeline_call=load_model_with_device, device_loader="device_map"
    )
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
