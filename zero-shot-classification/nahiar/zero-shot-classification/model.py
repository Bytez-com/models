from typing import List
import torch
from model_loader import load_model

pipe = load_model()


def model_run(text: str, candidate_labels: List[str], params: dict):
    """
    Runs model inference, using pipe

    required imports: transformers
    """
    return pipe(text, candidate_labels, **params)


def model_eject():
    """
    Saves the pytorch model to disk

    required imports: import torch
    """
    model_path = "model_weights.pth"

    torch.save(pipe.model.state_dict(), model_path)

    return model_path
