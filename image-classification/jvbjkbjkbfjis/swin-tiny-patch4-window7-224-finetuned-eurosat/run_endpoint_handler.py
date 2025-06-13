from flask import jsonify
from model import model_run


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    b64ImageBufferPng = request.json["b64ImageBufferPng"]

    results = model_run(b64ImageBufferPng, params)

    # Return the base64-encoded image in a JSON response
    return jsonify({"output": results})
