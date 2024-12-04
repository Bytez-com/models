import os
from dataclasses import dataclass
from multiprocessing import Process


@dataclass
class Test:
    model_type: str
    MODEL_ID: str
    TASK: str
    SUB_TASK: str
    DEVICE: str
    USE_TASK_MODEL_REGISTRY: str
    args_dict: dict
    kwargs: dict

    @property
    def args(self):
        args = self.args_dict.values()
        return args

    def set_environ(self):
        self.print_test_info()

        os.environ["DEVICE"] = self.DEVICE
        os.environ["TASK"] = self.TASK
        os.environ["SUB_TASK"] = self.SUB_TASK
        os.environ["MODEL_ID"] = self.MODEL_ID
        os.environ["USE_TASK_MODEL_REGISTRY"] = self.USE_TASK_MODEL_REGISTRY

    def print_test_info(self):
        model_types = [self.model_type, self.TASK]

        if self.SUB_TASK:
            model_types.append(self.SUB_TASK)

        model_types += [self.MODEL_ID]

        line = " -> ".join(model_types)

        print(f"Setting environment for test: {line}")


tests = [
    Test(
        model_type="transformers",
        MODEL_ID="openai-community/gpt2",
        DEVICE="AUTO",
        TASK="text-generation",
        SUB_TASK="",
        USE_TASK_MODEL_REGISTRY="false",
        args_dict={"text_inputs": "hello my name jeff"},
        kwargs={
            "max_new_tokens": 50,
        },
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="BUAADreamer/PaliGemma-3B-Chat-v0.2",
        DEVICE="AUTO",
        TASK="text-to-text-to-vision",
        SUB_TASK="chat",
        USE_TASK_MODEL_REGISTRY="true",
        args_dict={
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a friendly chatbot who responds in the tone of a pirate",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this image?"},
                        {
                            "type": "image_url",
                            # duck
                            "image_url": "https://hips.hearstapps.com/hmg-prod/images/how-to-keep-ducks-call-ducks-1615457181.jpg?crop=0.670xw:1.00xh;0.157xw,0&resize=980:*",
                        },
                    ],
                },
            ],
        },
        kwargs={
            "max_new_tokens": 50,
        },
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="google/paligemma-3b-mix-224",
        DEVICE="AUTO",
        TASK="text-to-text-to-vision",
        SUB_TASK="default",
        USE_TASK_MODEL_REGISTRY="true",
        args_dict={
            "text": "What is this?",
            "images": "https://images.squarespace-cdn.com/content/v1/54822a56e4b0b30bd821480c/45ed8ecf-0bb2-4e34-8fcf-624db47c43c8/Golden+Retrievers+dans+pet+care.jpeg?format=750w",
        },
        kwargs={
            "max_new_tokens": 50,
        },
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="meta-llama/Llama-3.2-11B-Vision",
        DEVICE="AUTO",
        TASK="text-to-text-to-vision",
        SUB_TASK="default",
        USE_TASK_MODEL_REGISTRY="true",
        args_dict={
            "text": "Haiku",
            "images": "https://images.squarespace-cdn.com/content/v1/54822a56e4b0b30bd821480c/45ed8ecf-0bb2-4e34-8fcf-624db47c43c8/Golden+Retrievers+dans+pet+care.jpeg?format=750w",
        },
        kwargs={
            "max_new_tokens": 10,
        },
    ),
]


for test in tests:
    test.set_environ()

    def run_test(run_as_child_process=False):
        def run_pipe():
            try:
                from model_loaders.get_model_loader import pipe
                from streamer import SingleTokenStreamer

                streamer = SingleTokenStreamer(tokenizer=pipe.tokenizer)

                result = pipe(streamer=streamer, *test.args, **test.kwargs)

                print("Result is: ", result)
            except Exception as exception:
                print(exception)
                return exception

        if run_as_child_process:

            process = Process(target=run_pipe)

            process.daemon = False

            process.start()

            process.join()

        else:
            from model_loaders.get_model_loader import pipe
            from streamer import SingleTokenStreamer

            streamer = SingleTokenStreamer(tokenizer=pipe.tokenizer)

            result = pipe(streamer=streamer, *test.args, **test.kwargs)
            print("Result is: ", result)

    run_test(run_as_child_process=True)

    a = 2

    pass
