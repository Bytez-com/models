ARG IMAGE_NAME
FROM $IMAGE_NAME:repo
#
ARG MODEL_ID
ENV MODEL_ID=$MODEL_ID
ARG TASK
ENV TASK=$TASK
ENV PORT=8002
ENV DISABLE_ANALYTICS=true
#
EXPOSE $PORT
# Copy /server to container /server
COPY ./server ./
# RUN pip install -r requirements.txt
# Run the server
ENTRYPOINT ["./serve.sh"]