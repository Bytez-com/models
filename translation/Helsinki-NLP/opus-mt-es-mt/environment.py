import os
import json
import torch

# if it hasn't already been set, prefer HF transfer
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

TASK = os.environ.get("TASK")

MODEL_ID = os.environ.get("MODEL_ID")

# purely for debug, useful when looking at container logs in production, helps make sense of model DL and load times
MODEL_SIZE_GB = float(os.environ.get("MODEL_SIZE_GB", "12345678"))

# controls conditional compatibility logic to allow models to work on CPU only machines
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# 8002 to avoid conflicts with testing locally againt local api server
PORT = os.environ.get("PORT", 8002)

# prevents calls to analytics when in the testing pipeline, or via instances api
DISABLE_ANALYTICS = json.loads(os.environ.get("DISABLE_ANALYTICS", "false"))

# Code is being run from gunicorn and not the debugger (flask will not start up)
START_FLASK_DEBUG_SERVER = json.loads(
    os.environ.get("START_FLASK_DEBUG_SERVER", "false")
)

USE_PRODUCTION_ANALYTICS_ENDPOINT = json.loads(
    os.environ.get("USE_PRODUCTION_ANALYTICS_ENDPOINT", "false")
)

# when a model, such as a text generation can produce output as it generates, this will enable such printing
MODEL_LOGGING = json.loads(os.environ.get("MODEL_LOGGING", "false"))

API_KEY = os.environ.get("KEY")

HF_API_KEY = os.environ.get("HF_API_KEY")

CONSTANTS_DICT = {
    "TASK": TASK,
    "MODEL": MODEL_ID,
    "DEVICE": DEVICE,
    "PORT": PORT,
    # For debugging containers we run, we disable analytics
    **(
        {
            "MODEL_LOGGING": MODEL_LOGGING,
            "MODEL_SIZE_GB": MODEL_SIZE_GB,
            "DISABLE_ANALYTICS": DISABLE_ANALYTICS,
            "START_FLASK_DEBUG_SERVER": START_FLASK_DEBUG_SERVER,
            "USE_PRODUCTION_ANALYTICS_ENDPOINT": USE_PRODUCTION_ANALYTICS_ENDPOINT,
            "HF_API_KEY": HF_API_KEY,
        }
        if DISABLE_ANALYTICS
        else {}
    ),
}

print("Environment: ")
for key, value in CONSTANTS_DICT.items():
    print(f"{key}: {value}")

if HF_API_KEY:
    from huggingface_hub import login

    try:
        login(HF_API_KEY)
    except Exception as exception:
        print("Could not log into HF, model may fail to load...")
