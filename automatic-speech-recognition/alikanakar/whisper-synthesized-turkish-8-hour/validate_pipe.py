from transformers import AutoTokenizer
from environment import MODEL_ID

kwargs = {
    ### params ###
}


def validate_pipe(pipe):
    # Load the tokenizer if it's not present, pipeline() doesn't always get this right
    if not pipe.tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
            pipe.tokenizer = tokenizer
        except Exception:
            print("Could not load tokenizer")
