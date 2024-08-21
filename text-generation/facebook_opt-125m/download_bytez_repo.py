import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))

OWNER = "Bytez-com"
REPO = "models"
BRANCH = "main"
TASK = os.environ.get("TASK")
IMAGE_NAME = os.environ.get("IMAGE_NAME")


def make_request(url):
    response = requests.get(url)

    # Check if the request was successful
    if not response.ok:
        raise Exception("Request failed")

    return response


# NOTE github doesn't support a way to directly get a dir's SHA without iterating through the entire contents of a directory
# so we opt for doing the recursion ourselves. This is ultimately less code
def get_model_files():
    files = _get_model_dir_tree(path=f"{TASK}/{IMAGE_NAME}")
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
            # Recurse into the directory
            sub_files = _get_model_dir_tree(sub_path)
            files.extend(sub_files)

    return files


def download_file(item):
    if item.get("type") != "file":
        raise Exception(
            "This should never happen, only files, not directories should be present"
        )

    path = item.get("path")

    print(f"Downloading: {path}")

    # this allows us to keep the path relative to our working directory, i.e. removes the task and model hierarchy from the path
    adjusted_path = path.split(f"{TASK}/{IMAGE_NAME}/")[1]

    file_path = f"{WORKING_DIR}/{adjusted_path}"
    # Create the directory structure if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    file_content = requests.get(item["download_url"]).content

    # Save the file locally
    with open(file_path, "wb") as file:
        file.write(file_content)


def download_files():
    model_files = get_model_files()

    with ThreadPoolExecutor() as executor:
        # Submit the download tasks to the executor
        futures = [executor.submit(download_file, item) for item in model_files]

        # Optional: wait for all downloads to complete
        for future in as_completed(futures):
            try:
                future.result()  # To raise any exceptions encountered during the execution
            except Exception as exc:
                print(f"An error occurred: {exc}")


def get_sha_for_dir(tree, path):
    for item in tree:
        if item["path"] == path:
            return item["sha"]

    raise Exception(f"Could not find: {path}")


if __name__ == "__main__":
    print("Downloading model repo...")

    download_files()

    print("Done dowloading model repo")
