import io
import base64
from flask import jsonify
from model import model_run


def run_endpoint_handler(request):
    params = request.json.get("params", {})

    images = model_run(params)

    image = images[0][0]

    # Convert the PIL image to a byte buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode the byte buffer to base64
    img_str = base64.b64encode(buffer.read()).decode("utf-8")

    # Return the base64-encoded image in a JSON response
    return jsonify({"output_png": img_str})
