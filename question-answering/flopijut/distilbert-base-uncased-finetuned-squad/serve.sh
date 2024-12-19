#!/bin/sh

pip install -r /server/requirements.txt
pip install flask[async]==3.0.3

# Use exec to replace the shell with gunicorn, passing environment variables directly into the command
exec gunicorn --bind 0.0.0.0:${PORT:-8000} --threads 8 --timeout 31536000 server:app