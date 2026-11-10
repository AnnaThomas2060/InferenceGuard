"""Standalone upload script to publish this RoBERTa model directory directly to Hugging Face Hub.
It reads configuration, weights, tokenizer, and model card files from the current folder.
The target repository is created if it does not already exist and updated with all assets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def push_folder_to_hub(repo_id: str, token: str | None = None, private: bool = False) -> None:
    """Upload all files from the current folder to the specified Hugging Face repository.
    The Hugging Face authentication token is retrieved from parameters or the environment.
    Repository creation is handled automatically prior to uploading folder contents."""
    hf_token = token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token is required. Pass --token or set HF_TOKEN environment variable.")

    current_dir = Path(__file__).resolve().parent
    api = HfApi(token=hf_token)

    print(f"Creating repository {repo_id} if needed...")
    create_repo(repo_id=repo_id, token=hf_token, private=private, exist_ok=True)

    print(f"Uploading files from {current_dir} to https://huggingface.co/{repo_id}...")
    api.upload_folder(
        folder_path=str(current_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Publish InferenceGuard RoBERTa risk classifier",
        ignore_patterns=["push_to_hub.py", "__pycache__/*", "*.pyc"],
    )
    print(f"Upload complete. Model is now accessible at https://huggingface.co/{repo_id}")


def main() -> None:
    """Parse command line arguments and execute the upload process.
    The repository identifier must be provided in the format username/repository-name.
    Optional flags allow setting repository visibility to private."""
    parser = argparse.ArgumentParser(description="Upload RoBERTa model files to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo identifier, e.g., user/model-name.")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token.")
    parser.add_argument("--private", action="store_true", help="Set repository to private.")
    args = parser.parse_args()

    push_folder_to_hub(repo_id=args.repo_id, token=args.token, private=args.private)


if __name__ == "__main__":
    main()
