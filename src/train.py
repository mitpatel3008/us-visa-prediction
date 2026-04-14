# src/train.py
"""
Model training script for US-Visa Prediction.
Trains XGBoost model and logs everything to MLflow.
"""

import mlflow
import mlflow.xgboost
import joblib
import argparse
import os
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from src.data import prepare_data

# Defaults
DEFAULT_PARAMS = {
    'n_estimators'     : 100,
    'max_depth'        : 6,
    'learning_rate'    : 0.3,
    'subsample'        : 1.0,
    'colsample_bytree' : 1.0,
    'eval_metric'      : 'logloss',
    'random_state'     : 42,
    'scale_pos_weight' : 2.0109
}

DEFAULT_THRESHOLD = 0.6

# Evaluate
def evaluate(y_true, y_pred, y_probs) -> dict:
    return {
        'accuracy' : float(accuracy_score(y_true, y_pred)),
        'f1'       : float(f1_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall'   : float(recall_score(y_true, y_pred)),
        'auc_roc'  : float(roc_auc_score(y_true, y_probs)),
    }

# Main Training Function
def train(
    data_path : str   = 'data/EasyVisa.csv',
    model_dir : str   = 'models/',
    threshold : float = DEFAULT_THRESHOLD,
    params    : dict  = None,
):
    params = params or dict(DEFAULT_PARAMS)
    os.makedirs(model_dir, exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("us-visa-prediction")

    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        # 1. Prepare Data
        print("Step 1: Preparing data...")
        X_train, X_test, y_train, y_test = prepare_data(data_path, model_dir)

        # 2. Log Parameters
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("data_path", data_path)

        # 3. Train Model
        print("Step 2: Training XGBoost model...")
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # 4. Predict with Custom Threshold
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_probs >= threshold).astype(int)

        # 5. Compute Metrics
        metrics = evaluate(y_test, y_pred, y_probs)
        mlflow.log_metrics(metrics)

        print("\n=== Results ===")
        for k, v in metrics.items():
            print(f"{k:12s}: {v:.4f}")
        print()

        print(classification_report(
            y_test,
            y_pred,
            target_names=["Denied", "Certified"]
        ))

        # 6. Save Artifacts
        joblib.dump(model,     f"{model_dir}/final_model.pkl")
        joblib.dump(threshold, f"{model_dir}/threshold.pkl")

        mlflow.xgboost.log_model(model, "model")
        mlflow.log_artifact(f"{model_dir}/preprocessor.pkl")
        mlflow.log_artifact(f"{model_dir}/threshold.pkl")

    print("Training complete. Model and artifacts saved.")
    return model, metrics

# CLI Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train US-Visa model")

    parser.add_argument("--data-path", default="data/EasyVisa.csv")
    parser.add_argument("--model-dir", default="models/")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.3)

    args = parser.parse_args()

    params = dict(DEFAULT_PARAMS)
    params.update({
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.lr
    })

    train(args.data_path, args.model_dir, args.threshold, params)