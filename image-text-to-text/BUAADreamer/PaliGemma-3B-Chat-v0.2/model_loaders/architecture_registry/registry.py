from collections import OrderedDict
import os
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoConfig
from environment import TASK, SUB_TASK, MODEL_ID, DEVICE
from model_loaders.architecture_registry.classes.model_entity import ModelEntity


WORKING_DIR = os.path.dirname(os.path.realpath(__file__))


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


def to_underscore_format(string: str):
    return string.replace("-", "_")


@dataclass
class Registry:
    @staticmethod
    def _get_model_entity(
        task: str,
        sub_task: str,
        model_id: str,
    ):
        task = to_underscore_format(task)
        sub_task = to_underscore_format(sub_task)

        # get the architecture
        config = AutoConfig.from_pretrained(model_id)
        model_architecture = config.architectures[0]

        # Specify the file path
        file_path = f"{WORKING_DIR}/tasks/{task}/sub_tasks/{sub_task}/architectures/{model_architecture}.py"

        # Get the module name from the file name
        module_name = Path(file_path).stem

        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # get the model entity from the module,
        # "model_cls" is a special variable name added to every ModelEntity sub class
        #  it acts as a uniform lookup key for each module
        model_entity_cls: ModelEntity = getattr(module, "model_cls")

        return model_entity_cls

    @staticmethod
    def get_model_entity(
        load_model=True,
        load_processor=True,
    ):

        print("Loading model...")

        # NOTE some models in the future, depending on their architecture may only support some sort of device configuration
        # if this is the case, we move the loading logic into the ModelEntry class for the particular architecture
        model_entity_cls = Registry._get_model_entity(
            task=TASK, sub_task=SUB_TASK, model_id=MODEL_ID
        )

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

                    kwargs = {}

                    # set the kwargs to specifically have the loading method and the device
                    kwargs.setdefault(loading_method, device)

                    model_entity = model_entity_cls.load_from_model_id(
                        model_id=MODEL_ID,
                        load_model=load_model,
                        load_processor=load_processor,
                        **kwargs,
                    )

                    pipe = model_entity

                    print(f"Loaded model via '{loading_method}' on device: ", device)

                    return pipe
                except Exception as exception:
                    collected_exception = exception

        raise collected_exception
