from dataclasses import dataclass
from enum import Enum, auto
import secrets
import string
import torch
import torch.nn.functional as F


class ComplianceFormat(Enum):
    DEFAULT = auto()
    OPEN_AI_COMPLETIONS = auto()
    OPEN_AI_CHAT_COMPLETIONS = auto()

    def is_openai_format(self) -> bool:
        return self in (
            ComplianceFormat.OPEN_AI_COMPLETIONS,
            ComplianceFormat.OPEN_AI_CHAT_COMPLETIONS,
        )

    @staticmethod
    def from_string(compliance_format: str):
        if compliance_format == "openai://completions":
            return ComplianceFormat.OPEN_AI_COMPLETIONS

        if compliance_format == "openai://chat/completions":
            return ComplianceFormat.OPEN_AI_CHAT_COMPLETIONS

        return ComplianceFormat.DEFAULT


def hf_sequence_logprobs_to_openai_chat_completions_format(
    sequence, scores, tokenizer, top_k=5
):
    # Only consider newly generated tokens (skip prompt part)
    generated_tokens = sequence[-len(scores) :]

    content_items = []

    for token_id, logit_scores in zip(generated_tokens, scores):
        result = hf_token_logprobs_to_openai_format(
            token_id, logit_scores, tokenizer, top_k
        )

        content_item = {
            "token": result["token"],
            "logprob": result["logprob"],
            "bytes": token_to_bytes(result["token"]),
            "top_logprobs": result["top_logprobs"],
            #
        }

        content_items.append(content_item)

    return {"content": content_items}


def hf_sequence_logprobs_to_openai_completions_format(
    sequence, scores, tokenizer, top_k=5
):
    # Only consider newly generated tokens (skip prompt part)
    generated_tokens = sequence[-len(scores) :]

    tokens = []
    token_logprobs = []
    top_logprobs = []
    text_offsets = []

    # Decode tokens one by one and track offsets
    running_text = ""

    for token_id, logit_scores in zip(generated_tokens, scores):
        result = hf_token_logprobs_to_openai_format(
            token_id, logit_scores, tokenizer, top_k
        )

        token = result["token"]
        logprob = result["logprob"]

        tokens.append(token)
        token_logprobs.append(logprob)

        # Format top_logprobs
        logprobs_dict = {lp["token"]: lp["logprob"] for lp in result["top_logprobs"]}

        top_logprobs.append(logprobs_dict)

        # --- text_offset calculation ---
        text_offsets.append(len(running_text))
        running_text += token

    return {
        "text_offset": text_offsets,
        "token_logprobs": token_logprobs,
        "tokens": tokens,
        "top_logprobs": top_logprobs,
    }


def hf_token_logprobs_to_openai_format(token_id, logit_scores, tokenizer, top_k=5):
    if logit_scores is None:
        return dict(token=None, logprob=None, top_logprobs=None)

    if not isinstance(logit_scores, torch.Tensor):
        logit_scores = torch.tensor(logit_scores).unsqueeze(0)

    log_probs = F.log_softmax(logit_scores, dim=-1)[0]  # shape: [vocab_size]

    # the token that was picked by highest probability
    token = tokenizer.decode(token_id)

    logprob = log_probs[token_id].item()

    # Top-k
    topk_logprobs, topk_indices = torch.topk(log_probs, top_k)

    top_logprobs = [
        {
            "token": tokenizer.decode(idx),
            "logprob": lp.item(),
            "bytes": token_to_bytes(tokenizer.decode(idx)),
        }
        for lp, idx in zip(topk_logprobs, topk_indices)
    ]

    filtered_logprobs = list(filter(log_prob_is_finite, top_logprobs))

    return dict(token=token, logprob=logprob, top_logprobs=filtered_logprobs)


def get_and_delete_from_dict(object: dict, key: str, fallback=None):
    if key in object:
        value = object[key]
        del object[key]

        return value

    return fallback


def get_and_delete_adaptation_kwargs(kwargs: dict):
    return dict(
        logprobs=get_and_delete_from_dict(kwargs, "logprobs"),
        top_logprobs=get_and_delete_from_dict(kwargs, "top_logprobs"),
        compliance_format=ComplianceFormat.from_string(
            get_and_delete_from_dict(kwargs, "compliance_format")
        ),
    )


def log_prob_is_finite(logprob_dict):
    logprob = logprob_dict["logprob"]
    return logprob != float("-inf") and logprob != float("inf")


def coallesce_infinity_to_none(value: float):
    if value == float("-inf") or value == float("inf"):
        return None
    return value


def generate_openai_id(prefix: str) -> str:
    rand = generate_id()
    return f"{prefix}-{rand}"


def generate_id():
    alphabet = string.ascii_letters + string.digits
    rand = "".join(secrets.choice(alphabet) for _ in range(24))
    return rand


def token_to_bytes(token: str):
    return list(token.encode("utf-8"))


def should_conform_to_input_expectations(
    compliance_format: ComplianceFormat, input: str
):
    if (
        compliance_format == ComplianceFormat.OPEN_AI_CHAT_COMPLETIONS
        and not isinstance(input, list)
    ):
        raise Exception(
            "When 'compliance_format' is 'openai://chat/completions' the prop 'text' must be a list of messages"
        )
    if compliance_format == ComplianceFormat.OPEN_AI_COMPLETIONS and not isinstance(
        input, str
    ):
        raise Exception(
            "When 'compliance_format' is 'openai://completions' the prop 'text' must be a string"
        )


SUPPORTED_COMPLETIONS_PARAMS = [
    "prompt",
    "suffix",
    "max_tokens",
    "temperature",
    "top_p",
    "n",
    "stream",
    "logprobs",
    "echo",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "best_of",
    "logit_bias",
    "user",
]


def oai_completions_req_to_hf_req(request):
    json: dict = request.json
    body = {}

    # things we just throw away
    get_and_delete_from_dict(json, "model")

    # Basic input
    body["text"] = get_and_delete_from_dict(json, "prompt")
    body["suffix"] = get_and_delete_from_dict(json, "suffix")
    body["stream"] = get_and_delete_from_dict(json, "stream")

    # Params mapped to HF
    params = dict(compliance_format="openai://completions")

    params["max_new_tokens"] = get_and_delete_from_dict(json, "max_tokens")
    params["temperature"] = get_and_delete_from_dict(json, "temperature")
    params["top_p"] = get_and_delete_from_dict(json, "top_p")
    params["n"] = get_and_delete_from_dict(json, "n")
    params["logprobs"] = get_and_delete_from_dict(json, "logprobs")
    params["echo"] = get_and_delete_from_dict(json, "echo")
    params["stop"] = get_and_delete_from_dict(json, "stop")
    params["presence_penalty"] = get_and_delete_from_dict(json, "presence_penalty")
    params["frequency_penalty"] = get_and_delete_from_dict(json, "frequency_penalty")
    params["best_of"] = get_and_delete_from_dict(json, "best_of")
    params["logit_bias"] = get_and_delete_from_dict(json, "logit_bias")
    params["user"] = get_and_delete_from_dict(json, "user")

    # check for unsupported params
    unsupported_params = [k for k in json.keys()]
    if unsupported_params:
        raise Exception(
            f"Unsupported params were passed to the API: {unsupported_params}, "
            f"Bytez only supports {SUPPORTED_COMPLETIONS_PARAMS} currently"
        )

    # clean up None values
    params = {k: v for k, v in params.items() if v is not None}

    body["params"] = params

    return AdaptedRequest(json=body)


SUPPORTED_CHAT_COMPLETIONS_PARAMS = [
    "messages",
    "stream",
    "max_tokens",
    "temperature",
    "top_p",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "functions",
    "function_call",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
]


def oai_chat_completions_req_to_hf_req(request):
    """
    Extracts parameters from an OpenAI-style chat/completions request
    and returns a dict with all the fields clearly separated.

    Args:
        request (dict): OpenAI chat completion request body.

    Returns:
        dict: Extracted parameters ready for further processing.
    """

    body = {}

    json: dict = request.json

    # things we just throw away
    get_and_delete_from_dict(json, "model")

    # basic inputs and controls
    body["text"] = get_and_delete_from_dict(json, "messages")
    body["stream"] = get_and_delete_from_dict(json, "stream")

    # params mapped to HF params, make sure compliance_format is specified
    params = dict(compliance_format="openai://chat/completions")

    params["max_new_tokens"] = get_and_delete_from_dict(json, "max_tokens")

    params["temperature"] = get_and_delete_from_dict(json, "temperature")
    params["top_p"] = get_and_delete_from_dict(json, "top_p")
    params["stop"] = get_and_delete_from_dict(json, "stop")

    params["presence_penalty"] = get_and_delete_from_dict(json, "presence_penalty")
    params["frequency_penalty"] = get_and_delete_from_dict(json, "frequency_penalty")
    params["logit_bias"] = get_and_delete_from_dict(json, "logit_bias")

    params["logprobs"] = get_and_delete_from_dict(json, "logprobs")
    params["top_logprobs"] = get_and_delete_from_dict(json, "top_logprobs")

    # Functions / tools
    params["functions"] = get_and_delete_from_dict(json, "functions")
    params["function_call"] = get_and_delete_from_dict(json, "function_call")
    params["tools"] = get_and_delete_from_dict(json, "tools")
    params["tool_choice"] = get_and_delete_from_dict(json, "tool_choice")
    params["parallel_tool_calls"] = get_and_delete_from_dict(
        json, "parallel_tool_calls"
    )

    # check for any params passed in that are not supported

    unsupported_params = []

    for key in json.keys():
        unsupported_params.append(key)

    if len(unsupported_params) > 0:
        raise Exception(
            f"Unsupported params were passed to the api: {unsupported_params}, Bytez only supports {SUPPORTED_CHAT_COMPLETIONS_PARAMS} currently"
        )

    items = list(params.items())

    # clean up the request
    for key, value in items:
        if value is None:
            del params[key]

    # Misc / experimental fields
    # params["response_format"] = get_and_delete_from_dict(json, "response_format")

    body["params"] = params

    return AdaptedRequest(json=body)


@dataclass
class AdaptedRequest:
    json: dict
