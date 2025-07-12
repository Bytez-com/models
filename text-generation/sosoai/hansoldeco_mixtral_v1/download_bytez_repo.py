import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))

OWNER = "Bytez-com"
REPO = "models"
BRANCH = "main"
TASK = os.environ.get("TASK")
MODEL_ID = os.environ.get("MODEL_ID")
USE_JSDELIVR = os.environ.get("USE_JSDELIVR", "false").lower() == "true"


def make_request(url):
    response = requests.get(url)
    if not response.ok:
        raise Exception("Request failed")
    return response


def get_model_files():
    files = _get_model_dir_tree(path=f"{TASK}/{MODEL_ID}")
    return files


def _get_model_dir_tree(path):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}"
    response = make_request(url=url)
    items = response.json()
    files = []

    for item in items:
        if item["type"] == "file":
            files.append(item)
        elif item["type"] == "dir":
            sub_path = item["path"]
            sub_files = _get_model_dir_tree(sub_path)
            files.extend(sub_files)

    return files


def get_jsdelivr_url(path):
    return f"https://cdn.jsdelivr.net/gh/{OWNER}/{REPO}@{BRANCH}/{path}"


def download_file(item):
    if item.get("type") != "file":
        raise Exception(
            "This should never happen, only files, not directories should be present"
        )

    path = item.get("path")

    adjusted_path = path.split(f"{TASK}/{MODEL_ID}/")[1]
    file_path = f"{WORKING_DIR}/{adjusted_path}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Use jsDelivr or GitHub download URL
    url = get_jsdelivr_url(path) if USE_JSDELIVR else item["download_url"]

    print(f"Downloading: {url}")

    file_content = requests.get(url).content

    with open(file_path, "wb") as file:
        file.write(file_content)


def download_files():
    model_files = get_model_files()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_file, item) for item in model_files]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"An error occurred: {exc}")


if __name__ == "__main__":
    print("Downloading model repo...")
    download_files()
    print("Done downloading model repo")
