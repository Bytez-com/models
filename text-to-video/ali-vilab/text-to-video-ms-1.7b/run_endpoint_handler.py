import base64
from flask import jsonify
from model import model_run
import imageio


def export_to_video(frames, output_video_path):
    writer = imageio.get_writer(
        output_video_path,
        fps=8,
        codec="libx264",  # ✅ Chrome-compatible
        ffmpeg_params=["-pix_fmt", "yuv420p"],  # ✅ required for browser playback
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


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

        print(f"Output b64 length is: {len(video_base64)}")

        with open("/tmp/video-decoded.mp4", "wb") as video_file:
            video_binary = base64.b64decode(video_base64)

            print(f"Output binary length is: {len(video_binary)}")

            video_file.write(video_binary)

        # Return response with base64 encoded video
        return jsonify({"output_mp4": video_base64})
    finally:
        # Ensure the temporary file is deleted
        # os.remove(tmp_file_path)
        pass
