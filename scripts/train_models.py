"""
Supervised Model Training & Evaluation Script for MazeMind AI.
Trains:
1. DecisionTreeClassifier (criterion="entropy", max_depth=5) -> predicts 'recommended_algo'
2. LinearRegression -> predicts 'astar_path_cost'
Saves serialized models (.pkl) and evaluation metrics (.json) to `models/`.
"""

import json
import os
import sys
from typing import Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

# Features used for training
FEATURE_COLUMNS = [
    "rows",
    "cols",
    "total_cells",
    "wall_density",
    "manhattan_dist",
    "euclidean_dist",
    "dead_end_count",
    "avg_open_neighbors",
]


def train_and_evaluate(
    data_path: str = "data/maze_benchmarks.csv",
    models_dir: str = "models",
    random_state: int = 42,
) -> Dict:
    """Trains both models, evaluates on hold-out test set, and exports artifacts."""
    print(f"Loading dataset from '{data_path}'...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_dataset.py first.")

    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # Integrity verification
    null_counts = df[FEATURE_COLUMNS + ["recommended_algo", "astar_path_cost"]].isnull().sum()
    print("Null check:")
    print(null_counts)
    if null_counts.sum() > 0:
        raise ValueError("Dataset contains unexpected null values!")

    X = df[FEATURE_COLUMNS]
    y_clf = df["recommended_algo"]
    y_reg = df["astar_path_cost"]

    # Train/Test Split (80/20)
    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X, y_clf, y_reg, test_size=0.20, random_state=random_state, stratify=y_clf
    )

    print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples.")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Train Decision Tree Classifier
    print("\n--- Training Model 1: Decision Tree Classifier ---")
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        random_state=random_state,
    )
    clf.fit(X_train, y_clf_train)

    y_clf_pred = clf.predict(X_test)
    clf_accuracy = float(accuracy_score(y_clf_test, y_clf_pred))
    classes = list(clf.classes_)
    cm = confusion_matrix(y_clf_test, y_clf_pred, labels=classes).tolist()
    clf_report = classification_report(y_clf_test, y_clf_pred, output_dict=True)

    feature_importances = {
        feat: round(float(imp), 4)
        for feat, imp in zip(FEATURE_COLUMNS, clf.feature_importances_)
    }

    print(f"Decision Tree Test Accuracy: {clf_accuracy:.4f}")
    print("Confusion Matrix:")
    print(np.array(cm))

    # 2. Train Linear Regression
    print("\n--- Training Model 2: Linear Regression ---")
    reg = LinearRegression()
    reg.fit(X_train, y_reg_train)

    y_reg_pred = reg.predict(X_test)
    r2 = float(r2_score(y_reg_test, y_reg_pred))
    mse = float(mean_squared_error(y_reg_test, y_reg_pred))
    rmse = float(np.sqrt(mse))

    coefficients = {
        feat: round(float(coef), 4)
        for feat, coef in zip(FEATURE_COLUMNS, reg.coef_)
    }
    intercept = round(float(reg.intercept_), 4)

    print(f"Linear Regression R2 Score: {r2:.4f}")
    print(f"Linear Regression MSE: {mse:.4f}")
    print(f"Linear Regression RMSE: {rmse:.4f}")

    # Save Models
    clf_path = os.path.join(models_dir, "decision_tree.pkl")
    reg_path = os.path.join(models_dir, "linear_regression.pkl")
    metrics_path = os.path.join(models_dir, "model_metrics.json")

    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    print(f"\nSaved models to '{clf_path}' and '{reg_path}'.")

    # Save Metrics
    metrics = {
        "dataset_rows": len(df),
        "test_size": len(X_test),
        "features": FEATURE_COLUMNS,
        "decision_tree": {
            "accuracy": round(clf_accuracy, 4),
            "classes": classes,
            "confusion_matrix": cm,
            "classification_report": clf_report,
            "feature_importances": feature_importances,
            "criterion": "entropy",
            "max_depth": 5,
        },
        "linear_regression": {
            "r2_score": round(r2, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "coefficients": coefficients,
            "intercept": intercept,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved evaluation metrics to '{metrics_path}'.")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
