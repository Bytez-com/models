# base image
FROM cuda_python:base

ARG MODEL_ID
ENV MODEL_ID=$MODEL_ID

ARG TASK
ENV TASK=$TASK

ENV PORT=80
# makes print() work with flask
ENV PYTHONUNBUFFERED=1
#
#
EXPOSE $PORT
# Copy entire contents of the build directory, has added benefit of acting as a backup
COPY ./ ./
RUN chmod +x serve.sh
#
# Run the server
ENTRYPOINT ["./serveDownloader.sh"]