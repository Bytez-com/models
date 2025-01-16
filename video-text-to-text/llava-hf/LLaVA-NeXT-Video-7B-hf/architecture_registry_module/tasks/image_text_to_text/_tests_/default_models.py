from architecture_registry_module.classes.test import Test

TASK = "image_text_to_text"
SUB_TASK = "default"

kwargs = {"max_new_tokens": 50}

tests = [
    # NOTE
    Test(
        model_type="architecture registry",
        MODEL_ID="google/paligemma-3b-mix-224",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="PaliGemmaForConditionalGeneration",
        args_dict={
            "text": "What kind of dog is this?",
            "images": "https://images.squarespace-cdn.com/content/v1/54822a56e4b0b30bd821480c/45ed8ecf-0bb2-4e34-8fcf-624db47c43c8/Golden+Retrievers+dans+pet+care.jpeg?format=750w",
        },
        kwargs=kwargs,
    ),
    # NOTE
    Test(
        model_type="architecture registry",
        MODEL_ID="meta-llama/Llama-3.2-11B-Vision",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="MllamaForConditionalGeneration",
        args_dict={
            "text": "Haiku",
            "images": "https://images.squarespace-cdn.com/content/v1/54822a56e4b0b30bd821480c/45ed8ecf-0bb2-4e34-8fcf-624db47c43c8/Golden+Retrievers+dans+pet+care.jpeg?format=750w",
        },
        kwargs=kwargs,
    ),
]
