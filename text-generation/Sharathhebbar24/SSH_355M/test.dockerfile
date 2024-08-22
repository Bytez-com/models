ARG IMAGE_NAME
FROM $IMAGE_NAME:repo
#
ARG MODEL_ID
ENV MODEL_ID=$MODEL_ID
ARG TASK
ENV TASK=$TASK
ENV PORT=8002
ENV DISABLE_ANALYTICS=true
# makes print() work with flask
ENV PYTHONUNBUFFERED=1
#
EXPOSE $PORT
# Copy /server to container /server
COPY ./server/testModelLoader.sh ./server/serve.sh ./

RUN chmod +x testModelLoader.sh

# RUN pip install -r requirements.txt
# Run the server
ENTRYPOINT ["./serve.sh"]