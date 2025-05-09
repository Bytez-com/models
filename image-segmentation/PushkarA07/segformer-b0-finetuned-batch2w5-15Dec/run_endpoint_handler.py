import io
import base64
from flask import jsonify
from model import model_run


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    b64ImageBufferPng = request.json["b64ImageBufferPng"]

    results = model_run(b64ImageBufferPng, params)

    formatted_results = []

    for item in results:
        label = item["label"]
        score = item["score"]
        image = item["mask"]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode the byte buffer to base64
        img_str = base64.b64encode(buffer.read()).decode("utf-8")

        formatted_result = {"label": label, "score": score, "mask_png": img_str}

        formatted_results.append(formatted_result)

    # Return the base64-encoded image in a JSON response
    return jsonify({"output": formatted_results})
