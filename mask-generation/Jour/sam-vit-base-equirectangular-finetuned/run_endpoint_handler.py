import base64
import io
from flask import jsonify
from PIL import Image
from model import model_run
from utils import convert_np


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    b64ImageBufferPng = request.json["b64ImageBufferPng"]

    results = model_run(b64ImageBufferPng, params)

    # get the dimensions of the input image
    image_data = base64.b64decode(b64ImageBufferPng)

    image = Image.open(io.BytesIO(image_data))

    width, height = image.size

    input_image_dimensions = {"width": width, "height": height}

    # format masks
    # these are 2d array with Bools, we'll convert them to 1's and 0's to reduce payload size
    formatted_masks = []

    for mask in convert_np(results["masks"]):
        new_mask = []

        for row in mask:
            new_row = []
            for column_pixel in row:
                new_row.append(int(column_pixel))

            new_mask.append(new_row)

        formatted_masks.append(new_mask)

    # format scores
    scores = results["scores"].tolist()

    # Return the base64-encoded image in a JSON response
    return jsonify(
        {
            "output": {
                "input_image_dimensions": input_image_dimensions,
                "masks": formatted_masks,
                "scores": scores,
            }
        }
    )
