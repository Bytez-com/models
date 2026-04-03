import torch
from model_loader import pipe


def model_run(text_input, images, params: dict):
    """
    Runs model inference, using pipe

    required imports: transformers
    """
    return pipe(text_input, images, **params)


def model_eject():
    """
    Saves the pytorch model to disk

    required imports: import torch
    """
    model_path = "model_weights.pth"

    torch.save(pipe.model.state_dict(), model_path)

    return model_path
