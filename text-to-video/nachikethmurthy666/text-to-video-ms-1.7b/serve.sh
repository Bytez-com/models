#!/bin/sh
python3 /server/download_bytez_repo.py

apt-get update && apt-get install -y libgl1 libglib2.0-0

pip install -r /server/requirements.txt

# Use exec to replace the shell with gunicorn, passing environment variables directly into the command
exec gunicorn --bind 0.0.0.0:${PORT:-8000} --timeout 31536000 server:app