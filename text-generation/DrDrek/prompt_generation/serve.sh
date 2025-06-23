#!/bin/sh
pip install -r /server/requirements.txt
# TODO remove this once all images are updated to include flask async pre installed
pip install flask[async]

# Use exec to replace the shell with gunicorn, passing environment variables directly into the command
exec gunicorn --bind 0.0.0.0:${PORT:-8000} --threads 32 --timeout 31536000 server:app