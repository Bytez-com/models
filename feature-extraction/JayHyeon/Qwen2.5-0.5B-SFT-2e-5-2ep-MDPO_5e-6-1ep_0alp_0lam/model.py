import torch
from model_loader import pipe


def model_run(text, params: dict):
    """
    Runs model inference, using pipe

    required imports: transformers
    """
    return pipe(text, **params)


def model_eject():
    """
    Saves the pytorch model to disk

    required imports: import torch
    """
    model_path = "model_weights.pth"

    torch.save(pipe.model.state_dict(), model_path)

    return model_path
