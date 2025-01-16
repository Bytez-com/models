from architecture_registry_module.classes.test import Test

TASK = "audio_text_to_text"
SUB_TASK = "chat"

args_dict = {
    "messages": [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3",
                },
                {"type": "text", "text": "What's that sound?"},
            ],
        },
    ],
}

kwargs = {
    "max_new_tokens": 50,
}

tests = [
    Test(
        model_type="architecture registry",
        MODEL_ID="Qwen/Qwen2-Audio-7B-Instruct",
        DEVICE="AUTO",
        TASK=TASK,
        SUB_TASK=SUB_TASK,
        ARCHITECTURE="Qwen2AudioForConditionalGeneration",
        args_dict=args_dict,
        kwargs=kwargs,
    ),
]
