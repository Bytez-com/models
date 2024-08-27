import io
import base64
from flask import jsonify
from scipy.io.wavfile import write as wav_write
from model import model_run


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    user_input = request.json["text"]

    # model inference
    model_output = model_run(user_input, params)

    # Create an in-memory buffer
    buffer = io.BytesIO()

    # Write the WAV file to the buffer
    try:
        # depending on the model, the 2d output array may be the num_samples, num_channels or num_channels, num_samples
        # we don't know upfront, this is simple and easy to do
        wav_write(
            buffer, rate=model_output["sampling_rate"], data=model_output["audio"]
        )
    except:
        wav_write(
            buffer,
            rate=model_output["sampling_rate"],
            # notice how this is transposed
            data=model_output["audio"].T,
        )

    # Get the WAV data from the buffer
    buffer.seek(0)
    wav_data = buffer.read()

    # Encode the WAV data to base64
    wav_base64 = base64.b64encode(wav_data).decode("utf-8")

    # Return response with base64 encoded audio
    return jsonify({"output_wav": wav_base64})
