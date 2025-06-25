from collections import OrderedDict
import os
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
import traceback
from transformers import PretrainedConfig
from environment import TASK, MODEL_ID, DEVICE, MODEL_LOADING_KWARGS
from architecture_registry_module.classes.model_entity import (
    ModelEntity,
)


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
        model_id: str,
    ):
        task = to_underscore_format(task)

        # get the architecture
        config = PretrainedConfig.get_config_dict(model_id)
        model_architecture = config[0]["architectures"][0]

        task_directory = f"{WORKING_DIR}/architecture_registry_module/tasks/{task}"

        # NOTE if there is a handler for a given architecture, this is what's loaded
        file_path = f"{task_directory}/architectures/{model_architecture}.py"

        # NOTE if there is a handler, but that handler needs multiple files and thus is held in a directory, e.g. ../image-text-to-text/MllamaForConditionalGeneration/MllamaForConditionalGeneration.py
        file_is_in_folder_path = f"{task_directory}/architectures/{model_architecture}/{model_architecture}.py"

        # NOTE if there is no handler for a given architecture, this is what's loaded
        # currently image-text-to-text loads models via pipe(), the idea is that you don't need a handler for every arch, some work out of the box
        fallback_module_path = f"{task_directory}/model_entity.py"

        # Get the module name from the file name
        module_name = Path(file_path).stem

        # we try each of these one at a time, it's a waterfall of defaulting
        modules_to_attempt_loading = [
            # custom specific handler for given architecture
            file_path,
            # custom specific handler for given architecture that
            file_is_in_folder_path,
            # fallback to the base class for a given task, e.g. image-text-to-text's base class attempts loading via pipeline()
            fallback_module_path,
        ]

        for path in modules_to_attempt_loading:
            print(f"\nAttempting to load module from path: {path}\n")
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # get the model entity from the module,
                # "model_cls" is a special variable name added to every ModelEntity sub class
                #  it acts as a uniform lookup key for each module
                model_entity_cls: ModelEntity = getattr(module, "model_cls")

                return model_entity_cls
            except:
                exception = traceback.format_exc()
                print(exception)
                pass

        raise Exception(exception)

    @staticmethod
    def get_model_entity(
        load_model=True,
        load_processor=True,
    ):

        print("Loading model...")

        # NOTE some models in the future, depending on their architecture may only support some sort of device configuration
        # if this is the case, we move the loading logic into the ModelEntry class for the particular architecture
        model_entity_cls = Registry._get_model_entity(task=TASK, model_id=MODEL_ID)

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

                    kwargs = {**MODEL_LOADING_KWARGS}

                    # set the kwargs to specifically have the loading method and the device
                    kwargs.setdefault(loading_method, device)

                    model_entity = model_entity_cls.load_from_model_id(
                        model_id=MODEL_ID,
                        load_model=load_model,
                        load_processor=load_processor,
                        **kwargs,
                    )

                    print(f"Loaded model via '{loading_method}' on device: ", device)

                    return model_entity
                except Exception as exception:
                    print(exception)
                    collected_exception = exception

        raise collected_exception


model_entity = Registry.get_model_entity()

pipe = model_entity
