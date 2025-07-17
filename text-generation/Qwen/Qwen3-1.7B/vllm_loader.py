import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine

# NOTE important, these make vLLM think it has EiB's worth of GPU memory
import vllm_mocks
import math
from dataclasses import dataclass
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.usage.usage_lib import UsageContext


# NOTE we lie to vllm and tell it it has EiB's worth of VRAM
# so this is semi arbitrary. However, internally vLLM says, you have 0.5EiB available, if it takes more than 0.5 EiB throw
GPU_MEMORY_UTILIZATION = 0.5


@dataclass
class PipeVLLM:
    engine: AsyncLLMEngine
    tokenizer: Any

    def __call__(self, request_input, **kwargs):
        prompt = (
            self.tokenizer.apply_chat_template(request_input, tokenize=False)
            if isinstance(request_input, list)
            else request_input
        )

        result = run_coroutine_sync(
            self.generate(request_input, prompt, **kwargs), timeout=60 * 10
        )

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

    async def generate(self, request_input, prompt, **kwargs):
        max_new_tokens = kwargs.get("max_new_tokens")

        streamer = kwargs.get("streamer")

        if streamer:
            del kwargs["streamer"]

        output_text = ""

        adapted_kwargs = self.adapt_hf_to_vllm_kwargs(**kwargs)

        self.engine.start_background_loop()

        stream = self.engine.generate(
            request_id="fake_req_id",
            prompt=prompt,
            sampling_params=SamplingParams(**adapted_kwargs),
        )

        async for out in stream:
            for output in out.outputs:
                text = output.text

                diff = text[len(output_text) :]
                output_text = text

                if streamer:
                    streamer.put(diff)

                if max_new_tokens and len(output.token_ids) >= max_new_tokens:
                    await stream.aclose()
                    break

        self.engine.shutdown_background_loop()

        if streamer:
            streamer.end()

        if isinstance(request_input, str):
            return [dict(generated_text=f"{prompt}{output_text}")]

        messages: list = request_input

        return [
            dict(
                generated_text=[
                    #
                    *messages,
                    dict(role="assistant", content=output_text),
                ]
            )
        ]


def load_model_with_vllm(model_id: str, torch_dtype, vllm_kwargs: dict):
    result = run_coroutine_sync(
        _load_model_with_vllm(model_id, torch_dtype, vllm_kwargs), timeout=60 * 10
    )

    return result


async def _load_model_with_vllm(
    model_id: str, torch_dtype, vllm_kwargs: dict
) -> PipeVLLM:
    max_model_len = vllm_kwargs.get("max_model_len", 4096)
    block_size = vllm_kwargs.get("block_size", 16)

    # NOTE this disables the profiled compilation of the cuda kernels
    enforce_eager = vllm_kwargs.get("enforce_eager", True)

    # NOTE It seems that it will always generate and not get stuck waiting for cache that isn't available as long as there is + 1 extra slot
    # which is presumably used for swapping or something to that effect. There may also be a bad conditional statement in their code
    num_blocks = math.ceil(max_model_len / block_size) + 1

    kwargs = {
        **dict(
            model=model_id,
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

    print("Loading vLLM...")

    engine_args = AsyncEngineArgs(
        **kwargs,
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.ENGINE_CONTEXT
    )

    tokenizer = await engine.get_tokenizer()

    print("vLLM loaded")

    return PipeVLLM(
        engine=engine,
        tokenizer=tokenizer,
    )


# this allows us to run async functions in sync code
def run_coroutine_sync(coroutine: Coroutine[Any, Any, Any], timeout: float = 30):
    def run_in_new_loop():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    if threading.current_thread() is threading.main_thread():
        if not loop.is_running():
            return loop.run_until_complete(coroutine)
        else:
            with ThreadPoolExecutor() as pool:
                future = pool.submit(run_in_new_loop)
                return future.result(timeout=timeout)
    else:
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()
