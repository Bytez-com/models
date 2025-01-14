from architecture_registry_module.classes.test import Test

TASK = "video_text_to_text"
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
                {"type": "text", "text": "Why is this video so funny?"},
                {
                    "type": "video",
                    # baby reading a book
                    "url": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                },
            ],
        },
    ],
}

kwargs = {"max_new_tokens": 50}

tests = [
    Test(
        model_type="architecture registry",
        MODEL_ID="llava-hf/LLaVA-NeXT-Video-7B-hf",
        DEVICE="AUTO",
        TASK="video_text_to_text",
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="LlavaNextVideoForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="xjtupanda/Idefics3-30K-mix-finetune",
        DEVICE="AUTO",
        TASK="video_text_to_text",
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="Idefics3ForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    Test(
        model_type="architecture registry",
        MODEL_ID="xjtupanda/Idefics3-200K-video-finetune",
        DEVICE="AUTO",
        TASK="video_text_to_text",
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="Idefics3ForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
    # NOTE this looks like it was yanked and requires too much handling for us to realistically feel confident in its robustness
    # Test(
    #     model_type="architecture registry",
    #     MODEL_ID="Sri-Vigneshwar-DJ/Apollo-LMMs-Apollo-1.5B-t32",
    #     DEVICE="AUTO",
    #     TASK="video_text_to_text",
    #     SUB_TASK=SUB_TASK,
    #     ARCHITECTURE="Idefics3ForConditionalGeneration",
    #     args_dict=args_dict,
    #     kwargs=kwargs,
    # ),
]
