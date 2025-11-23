import os
import pandas as pd
from datasets import Dataset
from huggingface_hub import login

HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not found")
login(token=HF_TOKEN)

# Load raw dataset
try:
    df = pd.read_csv("data/tourism.csv")
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

    # Create HuggingFace dataset
    dataset = Dataset.from_pandas(df)

    # Upload to HuggingFace
    dataset.push_to_hub(
        "arnavarpit/VUA-MLOPS",
        private=False,
        token=HF_TOKEN
    )
    print("Raw dataset uploaded to HuggingFace: arnavarpit/VUA-MLOPS")

except Exception as e:
    print(f"Error uploading dataset: {e}")
