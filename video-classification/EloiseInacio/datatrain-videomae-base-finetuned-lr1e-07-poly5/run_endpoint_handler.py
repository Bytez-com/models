import os
import base64
import tempfile
from flask import jsonify
from model import model_run


def run_endpoint_handler(request):
    b64VideoBufferMp4 = request.json.get("b64VideoBufferMp4")
    mp4Url = request.json.get("mp4Url")
    params = request.json.get("params", {})

    # NOTE short circuit if a download url is provided
    if mp4Url:
        print(
            f"mp4 was provided as url: {mp4Url}",
        )
        videos = [mp4Url]
        results = model_run(videos, params)
        return jsonify({"output": results})

    data = base64.b64decode(b64VideoBufferMp4)

    # Create a temporary file and write the data to it
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_file_path = temp_file.name

    try:
        temp_file.write(data)
        temp_file.close()  # Close the file to ensure data is written and file is available for reading

        videos = [temp_file_path]

        results = model_run(videos, params)

        # Return response with base64 encoded audio
        return jsonify({"output": results})
    finally:
        # Ensure the temporary file is deleted
        os.remove(temp_file_path)
