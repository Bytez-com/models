from flask import jsonify, Response

from utils import model_run_generator
from model import model_run


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    user_input = request.json["text"]
    stream = request.json.get("stream", False)

    if stream:
        output_generator = model_run_generator(user_input=user_input, params=params)

        return Response(
            output_generator(),
            content_type="text/event-stream; charset=utf-8",
        )

    # model inference
    model_output = model_run(user_input, params)

    # return response
    return jsonify({"output": model_output})
