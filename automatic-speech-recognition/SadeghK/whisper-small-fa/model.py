import torch
from model_loader import load_model

pipe = load_model()


def model_run(text, params: dict):
    """
    Runs model inference, using pipe

    required imports: transformers
    """

    forward_params: dict = params.get("forward_params")

    # NOTE you cannot pass in language as a normal param for these models
    # they must be part of the forward params
    if forward_params:
        pipe._forward_params = {
            #
            **pipe._forward_params,
            **forward_params,
        }

        del params["forward_params"]

    return pipe(text, **params)


def model_eject():
    """
    Saves the pytorch model to disk

    required imports: import torch
    """
    model_path = "model_weights.pth"

    torch.save(pipe.model.state_dict(), model_path)

    return model_path
