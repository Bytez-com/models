# base image
FROM cuda_python:base
#
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
# Copy /server to container /server
COPY ./server ./
RUN chmod +x serve.sh
#
# Run the server
ENTRYPOINT ["./serve.sh"]