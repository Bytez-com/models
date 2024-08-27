#!/bin/sh
apt update

apt install tesseract-ocr tesseract-ocr-eng libtesseract-dev -y

pip install -r /server/requirements.txt

# Use exec to replace the shell with gunicorn, passing environment variables directly into the command
exec gunicorn --bind 0.0.0.0:${PORT:-8000} --timeout 31536000 server:app