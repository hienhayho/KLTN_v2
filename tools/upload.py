import argparse
from huggingface_hub import upload_folder, create_repo

parser = argparse.ArgumentParser()
parser.add_argument(
    "--folder_path",
    type=str,
    required=True,
    help="Path to the folder you want to upload",
)
parser.add_argument(
    "--repo_id",
    type=str,
    required=True,
    help="Repository ID on Hugging Face Hub where the folder will be uploaded",
)
args = parser.parse_args()

# local_dir = "qwen2_5_1_5b_56k"
local_dir = args.folder_path
# repo_id = "hienhayho/qwen2.5-finetuned_56k"
repo_id = args.repo_id

create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True,
)

upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="model",
)
