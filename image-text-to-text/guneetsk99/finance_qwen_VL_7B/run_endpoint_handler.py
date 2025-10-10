import time
from flask import jsonify, Response

from environment import MODEL_ID
from utils import model_run_generator
from model import model_run, pipe
from model_loader import LOADED_ON_VLLM
import numpy as np
from adaptation import (
    ComplianceFormat,
    get_and_delete_adaptation_kwargs,
    hf_sequence_logprobs_to_openai_completions_format,
    hf_sequence_logprobs_to_openai_chat_completions_format,
    generate_openai_id,
    should_conform_to_input_expectations,
)

# allows us to extract sequence information from .generate()
if not LOADED_ON_VLLM:
    # only apply the monkey patch if we've loaded the model via pipeline()
    if pipe.pipe:
        og_generate = pipe.pipe._forward

        def monkey_patched__forward(model_inputs, generate_kwargs=None):
            generate_kwargs = {} if generate_kwargs is None else generate_kwargs
            prompt_text = model_inputs.pop("text")
            input_ids = (
                model_inputs["input_ids"]
                if "input_ids" in model_inputs
                else model_inputs["decoder_input_ids"]
            )  # for decoder-only models
            output = pipe.pipe.model.generate(
                **model_inputs,
                **{
                    **generate_kwargs,
                    "output_scores": True,
                    "return_dict_in_generate": True,
                }
            )

            return {
                "generated_sequence": output["sequences"][0],
                "scores": output["scores"],
                "prompt_text": prompt_text,
                "input_ids": input_ids,
            }

        pipe.pipe._forward = monkey_patched__forward

        og_postprocess = pipe.pipe.postprocess

        def monkey_patched_postprocess(model_outputs, **kwargs):
            output = og_postprocess(model_outputs, **kwargs)[0]
            sequence = model_outputs["generated_sequence"].tolist()
            output["sequence"] = sequence
            output["scores"] = model_outputs["scores"]
            # output["scores"] = [score.tolist() for score in model_outputs["scores"]]

            return [output]

        pipe.pipe.postprocess = monkey_patched_postprocess


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    text_input = request.json["text"]
    images = request.json.get("images")
    stream = request.json.get("stream", False)

    adaptation_kwargs = get_and_delete_adaptation_kwargs(params)

    compliance_format: ComplianceFormat = adaptation_kwargs["compliance_format"]

    should_conform_to_input_expectations(compliance_format, text_input)

    pipeline_kwargs = {
        **params,
        # these add props to the returned model output dict that are for computing logprobs and token usage
        "output_scores": True,
        "return_dict_in_generate": True,
    }

    vllm_kwargs = {**params, **adaptation_kwargs}

    params = vllm_kwargs if LOADED_ON_VLLM else pipeline_kwargs

    if stream:
        output_generator = model_run_generator(
            text_input,
            images,
            params=params,
            adaptation_kwargs=adaptation_kwargs,
        )

        return Response(
            output_generator(),
            content_type="text/event-stream; charset=utf-8",
        )

    if LOADED_ON_VLLM:
        model_output = model_run(
            text_input,
            images,
            params,
        )

        if compliance_format.is_openai_format():
            return jsonify(model_output)

        return jsonify({"output": model_output})

    should_get_logprobs = adaptation_kwargs["logprobs"]

    model_output = model_run(
        text_input,
        images,
        params,
    )

    model_output = model_output[0]

    usage = {
        "prompt_tokens": len(model_output["sequence"]) - len(model_output["scores"]),
        "total_tokens": len(model_output["sequence"]),
        "completion_tokens": len(model_output["scores"]),
        "prompt_tokens_details": None,
    }

    logprobs = (
        get_logprobs(adaptation_kwargs, text_input, model_output)
        if should_get_logprobs
        else None
    )

    model_output = clean_special_floats(model_output)

    if compliance_format == ComplianceFormat.OPEN_AI_COMPLETIONS:
        return jsonify(
            {
                "id": generate_openai_id("cmpl"),
                "object": "text_completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "text": model_output["generated_text"],
                        "logprobs": logprobs,
                        # TODO could benefit from knowing difference between length and stop
                        "finish_reason": "stop",
                        "stop_reason": None,
                    }
                ],
                "usage": usage,
            }
        )

    if compliance_format == ComplianceFormat.OPEN_AI_CHAT_COMPLETIONS:
        last_message = model_output["generated_text"][-1]
        return jsonify(
            {
                "id": generate_openai_id("chatcmpl"),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [
                    {
                        "message": {
                            # these by default are included in the response, if the model returns them, these will be overridden to a new value
                            "reasoning_content": None,
                            "tool_calls": [],
                            **last_message,
                            # in the event the model does not return "assistant" for role, we override it here
                            "role": "assistant",
                        },
                        "logprobs": logprobs,
                    }
                ],
                # TODO could benefit from knowing difference between length and stop
                "finish_reason": "stop",
                "stop_reason": None,
                "usage": usage,
            }
        )

    # clean up the output
    if "sequence" in model_output:
        del model_output["sequence"]

    if "scores" in model_output:
        del model_output["scores"]

    model_output["logprobs"] = logprobs
    model_output["usage"] = usage

    return jsonify({"output": [model_output]})


def clean_special_floats(data):
    """Recursively replace NaN, Infinity, and -Infinity with None."""
    if isinstance(data, dict):
        return {k: clean_special_floats(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_special_floats(v) for v in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None
    return data


def get_logprobs(
    adaptation_kwargs: dict,
    text_input,
    model_output,
):
    top_logprobs = adaptation_kwargs["top_logprobs"]

    sequence = model_output["sequence"]
    scores = model_output["scores"]

    if isinstance(text_input, str):
        return hf_sequence_logprobs_to_openai_completions_format(
            sequence, scores, pipe.tokenizer, top_logprobs
        )

    return hf_sequence_logprobs_to_openai_chat_completions_format(
        sequence, scores, pipe.tokenizer, top_logprobs
    )
