import io
import base64
from flask import jsonify
from model import model_run
from utils import convert_np

import numpy as np


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    b64ImageBufferPng = request.json["b64ImageBufferPng"]

    result = model_run(b64ImageBufferPng, params)

    # NOTE turn the depth_image into a b64 png string
    depth_image = result["depth"]

    buffer = io.BytesIO()
    depth_image.save(buffer, format="PNG")
    buffer.seek(0)

    img_str = base64.b64encode(buffer.read()).decode("utf-8")

    # NOTE turn the predicted_depth array into a bitmap
    predicted_depth = result["predicted_depth"]
    output = predicted_depth.squeeze().cpu().numpy()
    formatted_predicted_depth_array = (output * 255 / np.max(output)).astype("uint8")

    formatted_predicted_depth_array = convert_np(formatted_predicted_depth_array)

    # Return the base64-encoded image in a JSON response
    return jsonify(
        {
            "output": {
                "formatted_predicted_depth_array": formatted_predicted_depth_array,
                "depth_png": img_str,
            }
        }
    )
