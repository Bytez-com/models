import os
import json
import torch

# if it hasn't already been set, prefer HF transfer
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# always enable progress bars for dockerhub users
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = os.environ.get(
    "DISABLE_PROGRESS_BARS", "0"
)

TASK = os.environ.get("TASK")

MODEL_ID = os.environ.get("MODEL_ID")

# used for the model loading tracker
FILES_SIZE_GB = float(os.environ.get("FILES_SIZE_GB", "12345678"))

# used for the model loading tracker
MODEL_SIZE_GB = float(os.environ.get("MODEL_SIZE_GB", "12345678"))

# used for the model loading tracker
LOG_LOADING = bool(os.environ.get("LOG_LOADING", "false"))

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
    os.environ.get("USE_PRODUCTION_ANALYTICS_ENDPOINT", "true")
)

# when a model, such as a text generation can produce output as it generates, this will enable such printing
MODEL_LOGGING = json.loads(os.environ.get("MODEL_LOGGING", "false"))

API_KEY = os.environ.get("KEY")

HF_API_KEY = os.environ.get("HF_API_KEY")

SYSTEM_LOGS_PATH = os.environ.get("SYSTEM_LOGS_PATH", "/var/log/cloud-init-output.log")

DISABLE_PARALLEL_LOADING = json.loads(
    os.environ.get("DISABLE_PARALLEL_LOADING", "false")
)

HF_ENABLE_PARALLEL_DOWNLOADING = os.environ.get("HF_ENABLE_PARALLEL_DOWNLOADING")
HF_PARALLEL_DOWNLOADING_WORKERS = os.environ.get("HF_PARALLEL_DOWNLOADING_WORKERS")

DISABLE_PARALLEL_LOADING = os.environ.get("DISABLE_PARALLEL_LOADING")
PARALLEL_LOADING_WORKERS = os.environ.get("PARALLEL_LOADING_WORKERS")

# these are fed to models during model loading, they are used to provide primarily "variant" and "torch_dtype" props, can be used to provide overrides
MODEL_LOADING_KWARGS = json.loads(os.environ.get("MODEL_LOADING_KWARGS_JSON", "{}"))

TORCH_DTYPE = MODEL_LOADING_KWARGS.get("torch_dtype")

# if specified, we write the actual torch object to the property
if TORCH_DTYPE:
    MODEL_LOADING_KWARGS["torch_dtype"] = getattr(torch, TORCH_DTYPE)

CONSTANTS_DICT = {
    "TASK": TASK,
    "MODEL": MODEL_ID,
    "DEVICE": DEVICE,
    "PORT": PORT,
    "MODEL_LOADING_KWARGS": MODEL_LOADING_KWARGS,
    # For debugging containers we run, we disable analytics
    **(
        {
            "MODEL_LOGGING": MODEL_LOGGING,
            "FILES_SIZE_GB": FILES_SIZE_GB,
            "MODEL_SIZE_GB": MODEL_SIZE_GB,
            "LOG_LOADING": LOG_LOADING,
            "DISABLE_ANALYTICS": DISABLE_ANALYTICS,
            "START_FLASK_DEBUG_SERVER": START_FLASK_DEBUG_SERVER,
            "USE_PRODUCTION_ANALYTICS_ENDPOINT": USE_PRODUCTION_ANALYTICS_ENDPOINT,
            "HF_API_KEY": HF_API_KEY,
            "SYSTEM_LOGS_PATH": SYSTEM_LOGS_PATH,
            "HF_ENABLE_PARALLEL_DOWNLOADING": HF_ENABLE_PARALLEL_DOWNLOADING,
            "HF_PARALLEL_DOWNLOADING_WORKERS": HF_PARALLEL_DOWNLOADING_WORKERS,
            "DISABLE_PARALLEL_LOADING": DISABLE_PARALLEL_LOADING,
            "PARALLEL_LOADING_WORKERS": PARALLEL_LOADING_WORKERS,
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
