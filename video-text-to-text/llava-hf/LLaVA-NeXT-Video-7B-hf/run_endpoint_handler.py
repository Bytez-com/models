from flask import jsonify, Response

from utils import model_run_generator
from model import model_run


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    text_input = request.json["text"]
    videos = request.json.get("videos")
    stream = request.json.get("stream", False)

    if stream:
        output_generator = model_run_generator(text_input, videos, params=params)

        return Response(
            output_generator(),
            content_type="text/event-stream; charset=utf-8",
        )

    # model inference
    model_output = model_run(text_input, videos, params)

    # return response
    return jsonify({"output": model_output})
