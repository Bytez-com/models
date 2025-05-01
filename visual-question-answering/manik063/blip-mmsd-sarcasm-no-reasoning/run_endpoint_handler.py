from flask import jsonify
from model import model_run


def run_endpoint_handler(request):
    b64ImageBufferPng = request.json.get("b64ImageBufferPng")
    image_url = request.json.get("imageUrl")
    question = request.json.get("question")
    params = request.json.get("params", {})

    # NOTE short circuit if a download url is provided
    if image_url:
        print(
            f"Was provided url as input: {image_url}",
        )

        results = model_run(image_url, question, params)
        return jsonify({"output": results})

    results = model_run(b64ImageBufferPng, question, params)

    return jsonify({"output": results})
