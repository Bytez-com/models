ARG IMAGE_NAME
FROM $IMAGE_NAME:repo
#
ARG MODEL_ID
ENV MODEL_ID=$MODEL_ID
ARG TASK
ENV TASK=$TASK
ENV PORT=8002
ENV DISABLE_ANALYTICS=true
ENV MODEL_LOGGING=true
# makes print() work with flask
ENV PYTHONUNBUFFERED=1
#
EXPOSE $PORT
# Copy /server to container /server
COPY ./testModelLoader.sh ./serve.sh ./
RUN chmod +x testModelLoader.sh

# Run the server
ENTRYPOINT ["./testServe.sh"]