import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
from huggingface_hub import HfApi, login
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datasets import load_dataset
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not found")
login(token=HF_TOKEN)

# Setup MLflow
mlflow.set_experiment("tourism_package_prediction")

# Load train and test data
try:
    train_dataset = load_dataset("arnavarpit/VUA-MLOPS/train", split="train")
    train_df = train_dataset.to_pandas()
    test_dataset = load_dataset("arnavarpit/VUA-MLOPS/test", split="train")
    test_df = test_dataset.to_pandas()
    print("Data loaded from HuggingFace")

    # Convert any categorical columns back to numeric
    for col in train_df.columns:
        if train_df[col].dtype.name == 'category':
            train_df[col] = train_df[col].astype(int)
        if test_df[col].dtype.name == 'category':
            test_df[col] = test_df[col].astype(int)

except:
    train_df = pd.read_csv("data/train_data.csv")
    test_df = pd.read_csv("data/test_data.csv")
    print("Data loaded locally")

# Prepare features
X_train = train_df.drop(['CustomerID', 'ProdTaken'], axis=1)
y_train = train_df['ProdTaken']
X_test = test_df.drop(['CustomerID', 'ProdTaken'], axis=1)
y_test = test_df['ProdTaken']

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")

# Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    print(f"\n{model_name} Performance:")
    for metric, value in metrics.items():
        print(f"   {metric.capitalize()}: {value:.4f}")

    return metrics

# Train models with hyperparameter tuning
models_results = []

# 1. Decision Tree
print("Training Decision Tree...")
with mlflow.start_run(run_name="DecisionTree"):
    param_grid = {
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_dt = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_type", "DecisionTree")

    dt_metrics = evaluate_model(best_dt, X_test, y_test, "Decision Tree")
    mlflow.log_metrics(dt_metrics)
    mlflow.sklearn.log_model(best_dt, "model")

    models_results.append(("Decision Tree", best_dt, dt_metrics['roc_auc']))

# 2. Random Forest
print("Training Random Forest...")
with mlflow.start_run(run_name="RandomForest"):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_type", "RandomForest")

    rf_metrics = evaluate_model(best_rf, X_test, y_test, "Random Forest")
    mlflow.log_metrics(rf_metrics)
    mlflow.sklearn.log_model(best_rf, "model")

    models_results.append(("Random Forest", best_rf, rf_metrics['roc_auc']))

# 3. Gradient Boosting
print("Training Gradient Boosting...")
with mlflow.start_run(run_name="GradientBoosting"):
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 5, 7]
    }

    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_gb = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_type", "GradientBoosting")

    gb_metrics = evaluate_model(best_gb, X_test, y_test, "Gradient Boosting")
    mlflow.log_metrics(gb_metrics)
    mlflow.sklearn.log_model(best_gb, "model")

    models_results.append(("Gradient Boosting", best_gb, gb_metrics['roc_auc']))

# 4. XGBoost
print("Training XGBoost...")
with mlflow.start_run(run_name="XGBoost"):
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9]
    }

    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_type", "XGBoost")

    xgb_metrics = evaluate_model(best_xgb, X_test, y_test, "XGBoost")
    mlflow.log_metrics(xgb_metrics)
    mlflow.xgboost.log_model(best_xgb, "model")

    models_results.append(("XGBoost", best_xgb, xgb_metrics['roc_auc']))

# 5. AdaBoost
print("Training AdaBoost...")
with mlflow.start_run(run_name="AdaBoost"):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.5, 1.0, 1.5]
    }

    ada = AdaBoostClassifier(random_state=42)
    grid_search = GridSearchCV(ada, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_ada = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_type", "AdaBoost")

    ada_metrics = evaluate_model(best_ada, X_test, y_test, "AdaBoost")
    mlflow.log_metrics(ada_metrics)
    mlflow.sklearn.log_model(best_ada, "model")

    models_results.append(("AdaBoost", best_ada, ada_metrics['roc_auc']))

# Compare models and select best
print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)

results_df = pd.DataFrame([(name, score) for name, model, score in models_results],
                         columns=['Model', 'ROC_AUC'])
print(results_df)

# Find best model
best_model_name, best_model, best_score = max(models_results, key=lambda x: x[2])
print(f"\nBest Model: {best_model_name} (ROC-AUC: {best_score:.4f})")

# Save best model
os.makedirs("model_building", exist_ok=True)
joblib.dump(best_model, "model_building/best_model.joblib")

# Register best model to HuggingFace
api = HfApi()
repo_id = "arnavarpit/VUA-MLOPS/model"

try:
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj="model_building/best_model.joblib",
        path_in_repo="best_model.joblib",
        repo_id=repo_id,
        token=HF_TOKEN
    )
    print(f"Best model registered to HuggingFace: {repo_id}")
except Exception as e:
    print(f"Error registering model: {e}")
