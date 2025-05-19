from flask import jsonify
from model import model_run


def run_endpoint_handler(request):
    params = request.json.get("params", {})

    question = request.json["question"]
    context = request.json["context"]

    user_input = {"question": question, "context": context}

    # model inference
    model_output = model_run(user_input, params)

    # return response
    return jsonify({"output": model_output})
