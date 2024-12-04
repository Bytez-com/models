# base image
FROM cuda_python:base

ARG PYTHONPATH
ENV PYTHONPATH=$PYTHONPATH

ARG MODEL_ID
ENV MODEL_ID=$MODEL_ID

ARG TASK
ENV TASK=$TASK

ARG SUB_TASK
ENV SUB_TASK=$SUB_TASK

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