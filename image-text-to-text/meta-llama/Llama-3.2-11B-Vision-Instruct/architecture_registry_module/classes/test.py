import os
from dataclasses import dataclass


@dataclass
class Test:
    model_type: str
    MODEL_ID: str
    TASK: str
    SUB_TASK: str
    ARCHITECTURE: str
    DEVICE: str
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
        os.environ["MODEL_ID"] = self.MODEL_ID

    def print_test_info(self):
        model_types = [self.model_type, self.TASK, self.SUB_TASK, self.MODEL_ID, self.ARCHITECTURE, self.DEVICE]

        line = " -> ".join(model_types)

        print(f"Setting environment for test: {line}")