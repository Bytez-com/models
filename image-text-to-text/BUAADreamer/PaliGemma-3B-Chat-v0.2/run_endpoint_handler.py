from flask import jsonify, Response

from utils import model_run_generator
from model import model_run
import numpy as np


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    text_input = request.json["text_input"]
    images = request.json.get("images")
    stream = request.json.get("stream", False)

    if stream:
        output_generator = model_run_generator(text_input, images, params=params)

        return Response(
            output_generator(),
            content_type="text/event-stream; charset=utf-8",
        )

    # model inference
    model_output = model_run(text_input, images, params)

    model_output = clean_special_floats(model_output)

    # return response
    return jsonify({"output": model_output})


def clean_special_floats(data):
    """Recursively replace NaN, Infinity, and -Infinity with None."""
    if isinstance(data, dict):
        return {k: clean_special_floats(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_special_floats(v) for v in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None
    return data
