#!/usr/bin/env python3
"""
Deployment Script for Tourism Package Prediction Project
"""

import os

from huggingface_hub import HfApi, login

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not found")
login(token=HF_TOKEN)


def deploy_to_huggingface_space():
    """Deploy application to HuggingFace Spaces"""
    print("Deploying to HuggingFace Spaces...")

    try:
        api = HfApi()
        space_id = "arnavarpit/VUA-MLOPS"

        # First create the space if it doesn't exist
        try:
            api.create_repo(
                repo_id=space_id,
                repo_type="space",
                space_sdk="streamlit",
                exist_ok=True,
                private=False,
            )
            print(f"Space {space_id} created/verified")
        except Exception as e:
            print(f"Space creation note: {e}")

        files_to_upload = [
            ("app.py", "app.py"),
            ("requirements.txt", "requirements.txt"),
            ("Dockerfile", "Dockerfile"),
        ]

        # Add model file if it exists
        if os.path.exists("best_model.joblib"):
            files_to_upload.append(("best_model.joblib", "best_model.joblib"))
            print("Model file found and will be uploaded")
        else:
            print("Warning: Model file not found in deployment directory")

        print(f"Uploading files to space: {space_id}")

        for local_path, repo_path in files_to_upload:
            if os.path.exists(local_path):
                print(f"Uploading {local_path}...")
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=space_id,
                    repo_type="space",
                    token=HF_TOKEN,
                )
                print(f"{local_path} uploaded successfully")
            else:
                print(f"Warning: {local_path} not found, skipping...")

        print(f"\nDeployment completed!")
        print(f"App URL: https://huggingface.co/spaces/{space_id}")
        return True

    except Exception as e:
        print(f"Deployment error: {e}")
        return False


if __name__ == "__main__":
    success = deploy_to_huggingface_space()
    if success:
        print("✅ Deployment successful!")
    else:
        print("❌ Deployment failed!")
