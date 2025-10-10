import time
import threading
from model import model_run
from model import pipe
from model_loader import LOADED_ON_VLLM
from streamer import (
    SingleTokenStreamer,
    SingleTokenStreamerVllm,
)
import numpy as np
from environment import MODEL_ID


def model_run_generator(*args, params: dict, adaptation_kwargs: dict):

    if LOADED_ON_VLLM:
        # the compliance format is handed by the request handler itself for vLLM
        streamer = SingleTokenStreamerVllm()
    else:

        streamer = SingleTokenStreamer(
            tokenizer=pipe.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            model_id=MODEL_ID,
            **adaptation_kwargs,
        )

    params["streamer"] = streamer

    def model_run_thread():
        try:
            model_run(*args, params=params)
        except Exception as exception:
            streamer.text_queue.put(f"Error: {str(exception)}")

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
