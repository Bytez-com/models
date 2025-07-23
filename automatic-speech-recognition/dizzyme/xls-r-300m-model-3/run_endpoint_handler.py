import os
import base64
import tempfile
from flask import jsonify
from model import model_run
import librosa


def run_endpoint_handler(request):
    params = request.json.get("params", {})
    b64_audio_buffer_wav = request.json["b64AudioBufferWav"]

    # Decode the base64 audio buffer
    audio_data = base64.b64decode(b64_audio_buffer_wav)

    # Create a temporary file and write the audio data to it
    temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    try:
        temp_wav_file.write(audio_data)
        temp_wav_file_path = temp_wav_file.name
        temp_wav_file.close()  # Close the file to ensure data is written and file is available for reading

        audio_array, sample_rate = librosa.load(temp_wav_file_path, sr=None)

        audio_input = {"array": audio_array, "sampling_rate": sample_rate}

        # Model inference using the file path
        model_output = model_run(audio_input, params)

        # Return response with base64 encoded audio
        return jsonify({"output": model_output})
    finally:
        # Ensure the temporary file is deleted
        os.remove(temp_wav_file_path)
