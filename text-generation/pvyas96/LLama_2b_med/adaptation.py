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


def get_and_delete_from_dict(object: dict, key: str):
    if key in object:
        value = object[key]
        del object[key]

        return value

    return None


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
