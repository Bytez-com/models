import time
import threading
from model import model_run
from model import pipe
from streamer import SingleTokenStreamer
import numpy as np


def model_run_generator(user_input, params: dict):
    streamer: SingleTokenStreamer = pipe._forward_params.get("streamer")

    # add the streamer to the pipe if it isn't already there
    if not streamer:
        streamer = SingleTokenStreamer(
            tokenizer=pipe.tokenizer, skip_prompt=False, skip_special_tokens=True
        )

        pipe._forward_params = {**pipe._forward_params, "streamer": streamer}

    # always clear the stream before a request is made
    # if we introduce concurrency, then we cannot get away with a singleton
    streamer.reset()

    def model_run_thread():
        try:
            model_run(user_input, params=params)
        except Exception as exception:
            streamer.text_queue.put(
                'INTERNAL_BYTEZ_ERROR: arg "stream" was likely passed to a model that does not support streaming.'
            )
            # make sure the generator stops on an exception, otherwise the request will hang and never complete
            streamer.end()
            raise exception

    try:
        # run the model in its own thread, it will magically add
        # its streamed output to the streamer object's queue
        thread = threading.Thread(target=model_run_thread)

        # run the model
        thread.start()

        # this is our generator "hook" to the model
        def output_generator():
            for val in streamer:
                val: str

                yield val

            # cleanup the thread, needs to be here because flask doesn't have a way of cleaning this up otherwise
            thread.join()

        # return the generator
        return output_generator

    except Exception as exception:
        # always cleanup the thread on failure, no finally block because normal cleanup has to happen in the generator
        thread.join()
        raise exception


def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_np(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(element) for element in obj]
    else:
        return obj


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to execute.")
        return result

    return wrapper
