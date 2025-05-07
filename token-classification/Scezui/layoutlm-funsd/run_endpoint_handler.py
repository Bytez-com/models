from flask import jsonify
from model import model_run
from utils import convert_np


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    user_input = request.json["text"]

    # model inference
    model_output = model_run(user_input, params)

    # Ensure the model output is serializable
    model_output = convert_np(model_output)

    # return response
    return jsonify({"output": model_output})
