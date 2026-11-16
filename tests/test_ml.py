"""
Unit Tests for Machine Learning Pipeline and Dataset Integrity.
Verifies benchmark CSV schema, null counts, model deserialization,
and live inference output shapes and types.
"""

import os
import pytest
import numpy as np
import pandas as pd

from mazemind.environments import GridMaze
from mazemind.ml_predictor import MLPredictor, FEATURE_COLUMNS


def test_dataset_schema_and_integrity():
    """Verify data/maze_benchmarks.csv exists, has >= 1500 rows, and zero nulls."""
    csv_path = "data/maze_benchmarks.csv"
    assert os.path.exists(csv_path), f"Missing dataset at {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) >= 1500, f"Expected at least 1500 rows, got {len(df)}"

    # Required feature and target columns
    required_cols = FEATURE_COLUMNS + ["astar_path_cost", "astar_nodes_expanded", "recommended_algo"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Check for zero nulls
    assert df[required_cols].isnull().sum().sum() == 0, "Dataset contains null values"

    # Verify classes are valid
    valid_classes = {"A*", "BFS", "Greedy"}
    actual_classes = set(df["recommended_algo"].unique())
    assert actual_classes.issubset(valid_classes), f"Unexpected classes: {actual_classes}"


def test_ml_predictor_inference():
    """Verify MLPredictor loads saved models and produces valid predictions."""
    predictor = MLPredictor(models_dir="models")
    assert predictor.is_ready() is True

    # Test sample maze
    grid = np.zeros((15, 15), dtype=np.int32)
    grid[5, 2:8] = 1
    maze = GridMaze(grid=grid, start=(0, 0), goal=(14, 14))

    prediction = predictor.predict_maze(maze)

    assert "recommended_algo" in prediction
    assert prediction["recommended_algo"] in ["A*", "BFS", "Greedy"]

    assert "predicted_path_cost" in prediction
    assert isinstance(prediction["predicted_path_cost"], (float, int))
    assert prediction["predicted_path_cost"] > 0

    assert "probabilities" in prediction
    assert isinstance(prediction["probabilities"], dict)
    assert len(prediction["probabilities"]) > 0

    assert "features" in prediction
    for feat in FEATURE_COLUMNS:
        assert feat in prediction["features"]


def test_model_metrics_json():
    """Verify models/model_metrics.json has accuracy and R2 scores recorded."""
    metrics_path = "models/model_metrics.json"
    assert os.path.exists(metrics_path), f"Missing metrics file at {metrics_path}"

    predictor = MLPredictor(models_dir="models")
    metrics = predictor.metrics

    assert "decision_tree" in metrics
    assert "linear_regression" in metrics

    dt_metrics = metrics["decision_tree"]
    lr_metrics = metrics["linear_regression"]

    assert "accuracy" in dt_metrics
    assert 0.0 <= dt_metrics["accuracy"] <= 1.0

    assert "r2_score" in lr_metrics
    assert -1.0 <= lr_metrics["r2_score"] <= 1.0

    assert "rmse" in lr_metrics
    assert lr_metrics["rmse"] >= 0.0
