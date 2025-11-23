#!/usr/bin/env python3
"""
Data Preparation Script for Tourism Package Prediction
This script loads, cleans, preprocesses, and splits the data for model training.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, load_dataset
from huggingface_hub import login

def main():
    """Main function to prepare and upload processed datasets"""

    # HuggingFace authentication
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not found")

    try:
        login(token=HF_TOKEN)
        print("✅ Successfully authenticated with HuggingFace")
    except Exception as e:
        print(f"Warning: Failed to authenticate with HuggingFace: {e}")

    # Load data from HuggingFace
    try:
        dataset = load_dataset("arnavarpit/VUA-MLOPS", split="train")
        df = dataset.to_pandas()
        print(f"✅ Dataset loaded from HuggingFace: {len(df)} rows")
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        try:
            df = pd.read_csv("tourism.csv")
            print(f"✅ Dataset loaded locally: {len(df)} rows")
        except FileNotFoundError:
            raise FileNotFoundError("tourism.csv file not found. Please ensure the file is available.")

    # Data cleaning and preprocessing
    print("Starting data cleaning...")

    # Remove unnecessary columns
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # Handle missing values
    print("Missing values before cleaning:")
    missing_before = df.isnull().sum()[df.isnull().sum() > 0]
    if len(missing_before) > 0:
        print(missing_before)
    else:
        print("No missing values found")

    # Fill missing values for numerical columns
    numerical_cols = ['Age', 'DurationOfPitch', 'NumberOfFollowups', 'PreferredPropertyStar',
                     'NumberOfTrips', 'PitchSatisfactionScore', 'NumberOfChildrenVisiting', 'MonthlyIncome']

    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill missing values for categorical columns
    categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
                       'ProductPitched', 'Designation']

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

    # Fix data inconsistencies
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

    print("Missing values after cleaning:")
    missing_after = df.isnull().sum()[df.isnull().sum() > 0]
    if len(missing_after) > 0:
        print(missing_after)
    else:
        print("✅ No missing values remaining")

    # Feature engineering
    print("Feature engineering...")

    # Create income categories
    if 'MonthlyIncome' in df.columns:
        df['IncomeCategory'] = pd.cut(df['MonthlyIncome'],
                                     bins=[0, 15000, 25000, 35000, float('inf')],
                                     labels=[0, 1, 2, 3])  # Use numeric labels

    # Create age groups
    if 'Age' in df.columns:
        df['AgeGroup'] = pd.cut(df['Age'],
                               bins=[0, 25, 35, 45, 55, float('inf')],
                               labels=[0, 1, 2, 3, 4])  # Use numeric labels

    # Encode categorical variables
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        if col != 'CustomerID':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    print(f"✅ Data preprocessing completed! Final shape: {df.shape}")

    # Split data
    print("Splitting data...")
    X = df.drop(['CustomerID', 'ProdTaken'], axis=1)
    y = df['ProdTaken']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create train and test dataframes
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Add CustomerID
    train_df.insert(0, 'CustomerID', range(300000, 300000 + len(train_df)))
    test_df.insert(0, 'CustomerID', range(400000, 400000 + len(test_df)))

    print(f"✅ Data split completed!")
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # Upload to HuggingFace with error handling
    try:
        print("Uploading train dataset to HuggingFace...")
        train_dataset = Dataset.from_pandas(train_df)
        train_dataset.push_to_hub(
            "arnavarpit/VUA-MLOPS-train",
            private=False,
            token=HF_TOKEN
        )

        print("Uploading test dataset to HuggingFace...")
        test_dataset = Dataset.from_pandas(test_df)
        test_dataset.push_to_hub(
            "arnavarpit/VUA-MLOPS-test",
            private=False,
            token=HF_TOKEN
        )

        print("✅ Processed datasets uploaded to HuggingFace!")

    except Exception as e:
        print(f"❌ Failed to upload to HuggingFace: {e}")
        print("Continuing with local processing...")

    # Save locally as backup
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train_data.csv", index=False)
    test_df.to_csv("data/test_data.csv", index=False)
    print("✅ Data saved locally as backup")

    print("\n" + "="*50)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    main()
