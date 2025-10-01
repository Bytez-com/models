from typing import List, Dict, Union
from dataclasses import dataclass
import json
import math
import os
import subprocess
import time
import sys
import threading

import requests

PYTHON_EXECUTABLE = sys.executable


WORKING_DIR = os.path.dirname(os.path.realpath(__file__))

# NOTE we lie to vllm and tell it it has EiB's worth of VRAM
# so this is semi arbitrary. However, internally vLLM says, you have 0.5EiB available, if it takes more than 0.5 EiB throw
GPU_MEMORY_UTILIZATION = 0.5

BYTEZ_TO_OPEN_AI_CONTENT_ITEM_MAP = {
    "text": {
        "type": "text",
        "value_key": "text",
        "inner_value_fn": lambda value: value,
    },
    "image": {
        "type": "image_url",
        "value_key": "url",
        "inner_value_fn": lambda value: dict(url=value),
    },
    "audio": {
        "type": "audio_url",
        "value_key": "url",
        "inner_value_fn": lambda value: dict(url=value),
    },
    "video": {
        "type": "video_url",
        "value_key": "url",
        "inner_value_fn": lambda value: dict(url=value),
    },
    "document": None,
    "file": None,
}


@dataclass
class PipeVLLM:
    model_id: str
    port: int

    def __call__(self, *args, **kwargs):
        # note, this is to handle MM models, which take in images/audio/video as the second arg when pipe() is called
        # because everything is contained in the messages chain at args[0], we can safely ignore the second arg
        request_input = args[0]
        result = self.generate(request_input, **kwargs)
        return result

    def adapt_hf_to_vllm_kwargs(self, **kwargs) -> dict:
        hf_arg_to_vllm_map = dict(
            max_length="max_tokens",
            min_length="min_tokens",
            min_new_tokens="min_tokens",
            max_new_tokens="max_tokens",
            temperature="temperature",
            top_p="top_p",
            top_k="top_k",
            repetition_penalty="repetition_penalty",
            stop="stop",
            stop_token_ids="stop_token_ids",
            seed="seed",
            presence_penalty="presence_penalty",
            frequency_penalty="frequency_penalty",
            logit_bias="logit_bias",
            bad_words_ids="bad_words",
            ignore_eos="ignore_eos",
            num_return_sequences="n",  # maps to OpenAI-style n
        )

        new_kwargs = {}

        for key, value in kwargs.items():

            # this one is special, a registry pattern should be used if we need to increase the # of cases
            if key == "eos_token_id":
                if isinstance(value, int):
                    new_kwargs["stop_token_ids"] = [value]
                elif isinstance(value, list):
                    new_kwargs["stop_token_ids"] = value
                else:
                    raise TypeError("eos_token_id must be an int or list of ints")
                continue

            alias = hf_arg_to_vllm_map.get(key)

            if alias:
                # convert the kwarg to the vLLM equivalent for all other kwargs
                new_kwargs[alias] = value

            else:
                # if it's unsupported, we tell the user it's unsupported
                raise TypeError(f"{key} is not supported for this model on Bytez")

        return new_kwargs

    def generate(self, request_input, **kwargs):
        streamer = kwargs.get("streamer")

        if streamer:
            del kwargs["streamer"]

        adapted_kwargs = self.adapt_hf_to_vllm_kwargs(**kwargs)

        endpoint = ""

        is_string_input = isinstance(request_input, str)

        if is_string_input:
            adapted_kwargs["prompt"] = request_input
            endpoint = "/v1/completions"
        else:
            adapted_messages = self.adapt_bytez_input_to_openai_input(request_input)
            adapted_kwargs["messages"] = adapted_messages
            endpoint = "/v1/chat/completions"

        response = requests.post(
            f"http://localhost:{self.port}{endpoint}",
            headers={"Content-Type": "application/json"},
            json={
                "model": self.model_id,
                "stream": True,
                **adapted_kwargs,
            },
            # we always stream because we get the same result back
            # NOTE there may be performance downsides to this, albeit minor
            stream=True,
        )

        if not response.ok:
            error_obj = response.json()
            error_type = error_obj["type"]
            message = error_obj["message"]
            raise Exception(f"{error_type}: {message}")

        output_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.strip() == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    payload = json.loads(line[len("data: ") :])

                    if is_string_input:
                        delta = payload["choices"][0]["text"]

                    else:
                        delta = payload["choices"][0]["delta"].get("content", "")

                    output_text += delta

                    if streamer:
                        streamer.put(delta)

        if streamer:
            streamer.end()

        if is_string_input:
            return [dict(generated_text=f"{request_input}{output_text}")]

        messages: list = request_input

        return [
            dict(
                generated_text=[
                    *messages,
                    dict(role="assistant", content=output_text),
                ]
            )
        ]

    def adapt_bytez_input_to_openai_input(self, messages: List[str]):
        new_messages = []

        # always convert to lists of dicts for content for simplicity
        messages = self._adapt_string_only_content_to_lists(messages)

        for message in messages:
            role = message["role"]
            content: list = message["content"]

            new_content = []

            for content_item in content:
                type: Union[str, None] = content_item.get("type")

                if not type:
                    raise Exception("Prop `type` is not a string")

                content_item_map = BYTEZ_TO_OPEN_AI_CONTENT_ITEM_MAP[type]

                if not content_item_map:
                    raise Exception(f"Prop `{type}` is not supported")

                new_type = content_item_map["type"]

                value_key = content_item_map["value_key"]
                inner_value_fn = content_item_map["inner_value_fn"]

                value: Union[str, None] = content_item.get(value_key)

                if not value:
                    raise Exception(f"Prop `{value_key}` is not a string")

                formatted_value = inner_value_fn(value)

                new_content.append({"type": new_type, new_type: formatted_value})

            new_messages.append({"role": role, "content": new_content})

        return new_messages

    # "content": "The cat ran so fast"
    # becomes
    # "content": [{"type": "text", "text": "The cat ran so fast"}]
    def _adapt_string_only_content_to_lists(self, messages: List[Dict]):
        new_messages = []

        for message in messages:

            role = message.get("role")
            content = message.get("content")

            new_content = []

            if isinstance(content, str):
                new_content.append({"type": "text", "text": content})

            elif isinstance(content, dict):
                new_content.append(content)

            elif isinstance(content, list):

                new_content_items = []
                for content_item in content:
                    if isinstance(content_item, str):
                        new_content_items.append({"type": "text", "text": content_item})
                    elif isinstance(content_item, dict):
                        new_content_items.append(content_item)
                    else:
                        raise Exception(
                            "`content` can only contain strings or bytez content dicts"
                        )

                new_content += new_content_items
            else:
                raise Exception(
                    "`content` can only contain strings or bytez content dicts"
                )

            new_messages.append({"role": role, "content": new_content})

        return new_messages


def load_model_with_vllm(
    model_id: str, port: int, torch_dtype, vllm_kwargs: dict, vllm_env_vars: dict
) -> PipeVLLM:
    print("Starting vLLM server...")

    max_model_len = vllm_kwargs.get("max_model_len", 4096)
    block_size = vllm_kwargs.get("block_size", 16)

    # NOTE this disables the profiled compilation of the cuda kernels
    enforce_eager = vllm_kwargs.get("enforce_eager", True)

    # NOTE It seems that it will always generate and not get stuck waiting for cache that isn't available as long as there is + 1 extra slot
    # which is presumably used for swapping or something to that effect. There may also be a bad conditional statement in their code
    num_blocks = math.ceil(max_model_len / block_size) + 1

    args_dict = {
        **dict(
            model=model_id,
            port=port,
            dtype=torch_dtype if torch_dtype else "auto",
            max_model_len=max_model_len,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            num_gpu_blocks_override=num_blocks,
            block_size=block_size,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
        ),
        **vllm_kwargs,
    }

    # using PYTHON_EXECUTABLE allows the same version of python that launched this script to run the vLLM server
    # important with conda/venv, dockerfile already has deps installed, this provides flexibility while developing outside of the docker context
    args = [PYTHON_EXECUTABLE, f"{WORKING_DIR}/vllm_server.py"]
    if torch_dtype:
        args += ["--dtype", str(torch_dtype)]

    for key, value in args_dict.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args += [flag, str(value)]

    env = {**os.environ.copy(), **vllm_env_vars}

    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )

    def stream_logs(proc):
        for line in proc.stdout:
            print(line, end="", flush=True)

    threading.Thread(target=stream_logs, args=(process,), daemon=True).start()

    timeout = 60 * 10
    start = time.time()
    while True:
        # Check if process has exited
        retcode = process.poll()

        # NOTE vLLM currently always prints this out when it closes.
        # [rank0]:[W808 15:33:11.322491871 ProcessGroupNCCL.cpp:1476] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
        if retcode is not None:  # Process ended
            # Read any remaining logs
            leftover_logs = process.stdout.read()
            if leftover_logs:
                print(leftover_logs, end="", flush=True)
            raise RuntimeError(f"vLLM server process exited with code {retcode}")

        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.ok:
                break
        except Exception:
            pass

        if time.time() - start > timeout:
            process.terminate()
            raise TimeoutError("vLLM server failed to start in time.")

        time.sleep(1)

    print("vLLM server started.")

    return PipeVLLM(model_id=model_id, port=port)
