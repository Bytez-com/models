import threading
from model import model_run
from model import pipe
from streamer import SingleTokenStreamer


def model_run_generator(user_input, params: dict):
    # always have a fresh stream before a request is made
    # if we introduce concurrency, then we cannot get away with a singleton
    streamer: SingleTokenStreamer = pipe._forward_params.get("streamer")

    streamer.reset()

    def model_run_thread():
        model_run(user_input, params=params)

    try:
        # run the model in its own thread, it will magically add
        # its streamed output to the streamer object's queue
        thread = threading.Thread(target=model_run_thread)

        # this is our generator "hook" to the model
        def output_generator():
            for val in streamer:
                val: str

                yield val

            # cleanup the thread, needs to be here because flask doesn't have a way of cleaning this up otherwise
            thread.join()

        # run the model
        thread.start()

        # return the generator
        return output_generator

    except Exception as exception:
        # always cleanup the thread on failure, no finally block because normal cleanup has to happen in the generator
        thread.join()
        raise exception
