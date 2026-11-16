"""
ML Predictor Module for MazeMind AI.
Provides the `MLPredictor` class to load serialized models, run real-time inference,
and support dynamic re-training for the Streamlit UI.
"""

import json
import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from .environments import GridMaze

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


class MLPredictor:
    """
    Manages loading, inference, and evaluation for DecisionTree and LinearRegression models.
    """

    def __init__(
        self,
        models_dir: str = "models",
        metrics_file: str = "model_metrics.json",
    ):
        self.models_dir = models_dir
        self.metrics_path = os.path.join(models_dir, metrics_file)
        self.clf_path = os.path.join(models_dir, "decision_tree.pkl")
        self.reg_path = os.path.join(models_dir, "linear_regression.pkl")

        self.clf = None
        self.reg = None
        self.metrics: Dict[str, Any] = {}
        self.load_models()

    def is_ready(self) -> bool:
        """Returns True if models and metrics are loaded."""
        return self.clf is not None and self.reg is not None

    def load_models(self) -> bool:
        """Attempts to load serialized models and metrics."""
        if os.path.exists(self.clf_path) and os.path.exists(self.reg_path):
            self.clf = joblib.load(self.clf_path)
            self.reg = joblib.load(self.reg_path)

            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            return True
        return False

    def predict_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Runs inference given a dictionary of geometric features."""
        if not self.is_ready():
            raise RuntimeError("Models are not loaded. Train models before inference.")

        row_data = [features[col] for col in FEATURE_COLUMNS]
        df_feat = pd.DataFrame([row_data], columns=FEATURE_COLUMNS)

        rec_algo = str(self.clf.predict(df_feat)[0])
        probabilities = {}
        if hasattr(self.clf, "predict_proba"):
            probs = self.clf.predict_proba(df_feat)[0]
            for cls_name, p in zip(self.clf.classes_, probs):
                probabilities[str(cls_name)] = round(float(p), 4)

        pred_cost = float(self.reg.predict(df_feat)[0])
        pred_cost = max(1.0, round(pred_cost, 1))

        return {
            "recommended_algo": rec_algo,
            "probabilities": probabilities,
            "predicted_path_cost": pred_cost,
            "features": features,
        }

    def predict_maze(self, maze: GridMaze) -> Dict[str, Any]:
        """Extracts features from GridMaze and performs live inference."""
        features = maze.extract_features()
        return self.predict_features(features)
