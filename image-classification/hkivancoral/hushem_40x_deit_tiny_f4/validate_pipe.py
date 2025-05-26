from transformers import AutoImageProcessor
from environment import MODEL_ID

kwargs = {
    ### params ###
}


def validate_pipe(pipe):
    if not pipe.image_processor:
        try:
            processor = AutoImageProcessor.from_pretrained(MODEL_ID, **kwargs)
            pipe.image_processor = processor
        except Exception as exception:
            print("Could not load image_processor")
