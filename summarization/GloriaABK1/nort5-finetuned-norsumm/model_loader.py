from collections import OrderedDict
from transformers import pipeline
from environment import MODEL_ID, TASK, DEVICE, MODEL_LOADING_KWARGS
from validate_pipe import validate_pipe


# construct as a set to dedupe, then turn into list
FALL_BACK_DEVICES = [
    "auto",
    # if the device was specified, i.e. cuda for instances, then if auto fails, this will run
    DEVICE,
    # always fallback to cpu worst case scenario
    "cpu",
]

# Use OrderedDict to maintain order while deduplicating
DEVICES = list(OrderedDict.fromkeys(FALL_BACK_DEVICES))
DEVICES_NO_AUTO = DEVICES[1:]


DEFAULT_KWARGS = {
    ### params ###
    "task": TASK,
    "model": MODEL_ID,
    "trust_remote_code": True
}


def try_loading():
    # we'll try loading on "device_map" first, then "device". This is to ensure a model at least runs on the CPU if
    # it fails to load on cuda on an instance
    loading_methods = [
        ["device_map", DEVICES],
        # NOTE device is special, it doesn't support 'auto'
        ["device", DEVICES_NO_AUTO],
    ]

    collected_exception = None

    for loading_method, devices in loading_methods:
        for device in devices:
            try:
                print(
                    f"Attempting to load model via '{loading_method}' with device: ",
                    device,
                )

                kwargs = {**DEFAULT_KWARGS, **MODEL_LOADING_KWARGS}

                # set the kwargs to specifically have the loading method and the device
                kwargs.setdefault(loading_method, device)

                pipe = pipeline(**kwargs)

                print(f"Loaded model via '{loading_method}' on device: ", device)

                return pipe
            except Exception as exception:
                collected_exception = exception

    raise collected_exception


pipe = try_loading()

# this does a double check for things that should be present, e.g. tokenizers, image_processors, etc.
validate_pipe(pipe)

print("Model loaded")


def load_model():
    # instructions: do not modify this function
    # export the model loaded into global memory
    global pipe

    return pipe


if __name__ == "__main__":
    load_model()
