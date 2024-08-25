from flask import jsonify
from model import model_run
import numpy as np


def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_np(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(element) for element in obj]
    else:
        return obj


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    user_input = request.json["text"]

    # model inference
    model_output = model_run(user_input, params)

    # Ensure the model output is serializable
    model_output = convert_np(model_output)

    # return response
    return jsonify({"output": model_output})
