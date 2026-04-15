import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Import same pipeline used in training
from src.data import (
    load_data,
    drop_columns,
    clean_data,
    engineer_features,
    encode_target
)

# Load Artifacts
def load_artifacts(model_dir: str = "models/") -> dict:
    required = {
        "model": "final_model.pkl",
        "preprocessor": "preprocessor.pkl",
        "threshold": "threshold.pkl",
    }

    artifacts = {}
    for key, filename in required.items():
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{filename} not found in {model_dir}")
        artifacts[key] = joblib.load(path)

    print(f"Artifacts loaded from '{model_dir}'")
    print(f"Threshold: {artifacts['threshold']}")
    return artifacts

# Evaluation Function
def evaluate_model(
    model_dir="models/",
    X_test=None,
    y_test=None,
    save_plots=True,
    plots_dir="reports/",
):
    if X_test is None or y_test is None:
        raise ValueError("X_test and y_test cannot be None")

    artifacts = load_artifacts(model_dir)
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    threshold = float(artifacts["threshold"])

    # Apply preprocessing
    X_test_proc = preprocessor.transform(X_test)

    # Predictions
    y_probs = model.predict_proba(X_test_proc)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_probs)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"AUC-ROC  : {auc_roc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig1, ax1 = plt.subplots()
    disp.plot(ax=ax1)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr)
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.set_title("ROC Curve")

    # Save or show plots
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)
        fig1.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
        fig2.savefig(os.path.join(plots_dir, "roc_curve.png"))
    else:
        plt.show()

    # Close plots (important)
    plt.close(fig1)
    plt.close(fig2)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
    }

# Main execution
if __name__ == "__main__":
    print("Loading and preparing data...")

    # Apply same preprocessing pipeline
    df = load_data("data/EasyVisa.csv")
    df = drop_columns(df)
    df = clean_data(df)
    df = engineer_features(df)

    # Get features and target
    X, y = encode_target(df)

    print(f"Total samples: {len(df)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Test samples: {len(X_test)}")

    # Evaluate model
    print("\nRunning evaluation...\n")

    metrics = evaluate_model(
        model_dir="models/",
        X_test=X_test,
        y_test=y_test,
        save_plots=True
    )

    print("\n--- FINAL METRICS ---")
    print(metrics)