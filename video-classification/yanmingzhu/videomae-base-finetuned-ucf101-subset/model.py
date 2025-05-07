import torch
from model_loader import load_model

pipe = load_model()


def model_run(videos: list, params: dict):
    """
    Runs model inference, using pipe

    required imports: transformers
    """
    return pipe(videos=videos, **params)


def model_eject():
    """
    Saves the pytorch model to disk

    required imports: import torch
    """
    model_path = "model_weights.pth"

    torch.save(pipe.model.state_dict(), model_path)

    return model_path
