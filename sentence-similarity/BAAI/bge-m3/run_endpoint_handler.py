from flask import jsonify, Response
from utils import model_run_generator
from model import model_run
import numpy as np


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    user_input = request.json["text"]

    # model inference
    model_output = model_run(user_input, params)

    # Ensure the model output is serializable
    if isinstance(model_output, np.ndarray):
        model_output = model_output.tolist()

    # return response
    return jsonify({"output": model_output})
