from architecture_registry_module.classes.test import Test

TASK = "image_text_to_text"
SUB_TASK = "chat"

args_dict = {
    "messages": [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a friendly chatbot who responds in the tone of a pirate",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    #
                    "type": "text",
                    "text": "Describe the image.",
                },
                {
                    "type": "image",
                    # flower
                    "url": "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg",
                },
            ],
        },
    ],
}

kwargs = {"max_new_tokens": 50}

# NOTE if the architecture is not in the list below, it is not guaranteed to be supported

tests = [
    # NOTE
    Test(
        model_type="architecture registry",
        MODEL_ID="HuggingFaceTB/SmolVLM-Base",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="Idefics3ForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    # NOTE
    Test(
        model_type="architecture registry",
        MODEL_ID="llava-hf/llava-1.5-7b-hf",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="LlavaForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="Qwen/Qwen2-VL-2B-Instruct",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="Qwen2VLForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="prithivMLmods/Qwen2-VL-Math-Prase-2B-Instruct",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="Qwen2VLForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="Lemorra/Qwen2-VL",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="Qwen2VLForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="fredaddy/Qwen-VL-7B-2",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="Qwen2VLForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="MissFlash/qwen2-7b-instruct-amazon-description-merged",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="Qwen2VLForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="andrewhinh/qwen2-vl-7b-instruct-full-sft",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="Qwen2VLForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="llava-hf/llava-v1.6-vicuna-7b-hf",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="LlavaNextForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="LlavaOnevisionForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="lamm-mit/Cephalo-Idefics-2-vision-8b-alpha",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="Idefics2ForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    # NOTE works
    Test(
        model_type="architecture registry",
        MODEL_ID="giobin/idefics3_random_connector",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="Idefics3ForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    # NOTE
    Test(
        model_type="architecture registry",
        MODEL_ID="meta-llama/Llama-3.2-11B-Vision-Instruct",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="MllamaForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    # NOTE
    Test(
        model_type="architecture registry",
        MODEL_ID="BUAADreamer/PaliGemma-3B-Chat-v0.2",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="PaliGemmaForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    # NOTE
    Test(
        model_type="architecture registry",
        MODEL_ID="llava-hf/vip-llava-7b-hf",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="VipLlavaForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    )
]
