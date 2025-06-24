from flask import jsonify
from model import model_run


def run_endpoint_handler(request):
    b64ImageBufferPng = request.json["b64ImageBufferPng"]
    candidate_labels = request.json["candidate_labels"]
    params = request.json.get("params", {})

    results = model_run(b64ImageBufferPng, candidate_labels, params)

    # Return the base64-encoded image in a JSON response
    return jsonify({"output": results})
