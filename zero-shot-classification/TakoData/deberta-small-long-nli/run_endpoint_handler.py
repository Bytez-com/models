from flask import jsonify
from model import model_run


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    user_input = request.json["text"]
    candidate_labels = request.json["candidate_labels"]

    # model inference
    model_output = model_run(user_input, candidate_labels, params)

    # return response
    return jsonify({"output": model_output})
