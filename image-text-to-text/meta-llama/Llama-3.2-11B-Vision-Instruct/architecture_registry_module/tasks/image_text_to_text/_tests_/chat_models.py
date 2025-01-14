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
    # NOTE Has a bug where multiple gpus throw an index error because not all tensors are on the same gpu
    # come back to this and see if there's a work around
    # Test(
    #     model_type="architecture registry",
    #     MODEL_ID="Qwen/Qwen2-VL-2B-Instruct",
    #     DEVICE="AUTO",
    #     TASK=TASK,
    #     SUB_TASK=SUB_TASK,
    #     ARCHITECTURE="Qwen2VLForConditionalGeneration",
    #     args_dict=args_dict,
    #     kwargs=kwargs,
    # ),
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
    ),
    # NOTE this is a special example of a video_text_to_text model being classified as image_text_to_text
    Test(
        model_type="architecture registry",
        MODEL_ID="llava-hf/LLaVA-NeXT-Video-7B-32K-hf",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="LlavaNextVideoForConditionalGeneration",
        args_dict={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Why is this video so funny?"},
                        {
                            "type": "video",
                            # baby reading a book
                            "url": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                        },
                    ],
                },
            ],
        },
        kwargs=kwargs,
    ),
]
