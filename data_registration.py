#!/usr/bin/env python3
"""
Data Registration Script for Tourism Package Prediction
This script uploads the raw dataset to HuggingFace Hub for version control and accessibility.
"""

import pandas as pd
import os
from datasets import Dataset
from huggingface_hub import login

def main():
    """Main function to register dataset to HuggingFace Hub"""

    # HuggingFace authentication
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not found")

    try:
        login(token=HF_TOKEN)
        print("✅ Successfully authenticated with HuggingFace")
    except Exception as e:
        raise Exception(f"Failed to authenticate with HuggingFace: {e}")

    # Load the raw dataset
    try:
        df = pd.read_csv("tourism.csv")
        print(f"✅ Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        raise FileNotFoundError("tourism.csv file not found. Please ensure the file is in the repository root.")

    # Basic data info
    print("\nDataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Convert to HuggingFace dataset and upload
    try:
        dataset = Dataset.from_pandas(df)

        # Upload to HuggingFace Hub
        dataset.push_to_hub(
            "arnavarpit/VUA-MLOPS",
            private=False,
            token=HF_TOKEN
        )

        print("✅ Raw dataset successfully uploaded to HuggingFace Hub: arnavarpit/VUA-MLOPS")

    except Exception as e:
        raise Exception(f"Failed to upload dataset to HuggingFace Hub: {e}")

if __name__ == "__main__":
    main()
