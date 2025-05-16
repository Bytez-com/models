import os
import base64
from flask import jsonify
from model import model_run
from diffusers.utils import export_to_video


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    prompt = request.json["prompt"]
    negative_prompt = request.json.get("negativePrompt")

    # Create a temporary file and write the video data to it
    tmp_file_path = "/tmp/video.mp4"

    try:
        # Model inference using the file path
        video_frames = model_run(prompt, negative_prompt, params)
        video_frames = video_frames.squeeze()

        export_to_video(video_frames, output_video_path=tmp_file_path)

        # Read the video file and encode it as base64
        with open(tmp_file_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

        # Return response with base64 encoded video
        return jsonify({"output_mp4": video_base64})
    finally:
        # Ensure the temporary file is deleted
        os.remove(tmp_file_path)
