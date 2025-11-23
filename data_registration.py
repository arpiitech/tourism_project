from huggingface_hub import login, HfApi
from datasets import Dataset
import pandas as pd
import os

HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not found")
login(token=HF_TOKEN)

# Load the dataset
df = pd.read_csv("data/tourism.csv")
print(f"Dataset loaded: {len(df)} rows")
print("Dataset shape:", df.shape)
print("\nDataset columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# Create dataset object
dataset = Dataset.from_pandas(df)

# Upload to HuggingFace Hub
repo_id = "arnavarpit/VUA-MLOPS"
dataset.push_to_hub(
    repo_id,
    private=False,
    token=HF_TOKEN
)

print(f"You can access it at: https://huggingface.co/datasets/{repo_id}")
print(f"\nDataset uploaded successfully to: {repo_id}")
