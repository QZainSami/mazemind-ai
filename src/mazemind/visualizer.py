"""
Visualization Utilities for MazeMind AI.
Builds Matplotlib figures for:
- 2D Grid Maze (walls, exploration progression, solution paths)
- 8-Puzzle tile board state
- GA fitness convergence curves
- Confusion matrix heatmaps & Decision tree graphs
- Linear regression actual vs. predicted plots & correlation heatmaps
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for Linux servers, Docker & Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree

from .environments import GridMaze, EightPuzzle
from .search_agents import SearchResult


def plot_maze(
    maze: GridMaze,
    result: Optional[SearchResult] = None,
    step: Optional[int] = None,
    title: str = "Grid Maze",
    figsize: Tuple[int, int] = (7, 7),
) -> plt.Figure:
    """
    Renders a 2D GridMaze with optional exploration overlay and path.
    - If `step` is provided, shows exploration up to `step`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    rows, cols = maze.rows, maze.cols

    # Base display grid: 0 = Open (white), 1 = Wall (dark), 2 = Explored (cyan), 3 = Path (amber)
    display_grid = np.zeros((rows, cols), dtype=np.float32)
    display_grid[maze.grid == 1] = 1.0

    # Overlay exploration history
    if result and result.exploration_history:
        history_slice = (
            result.exploration_history[:step]
            if step is not None
            else result.exploration_history
        )
        for r, c in history_slice:
            if (r, c) != maze.start and (r, c) != maze.goal and maze.grid[r, c] == 0:
                display_grid[r, c] = 0.4  # Explored tint

    # Overlay final path if search was complete or step covers whole exploration
    show_path = (
        result
        and result.found
        and (step is None or step >= len(result.exploration_history))
    )
    if show_path and result.path:
        for r, c in result.path:
            if (r, c) != maze.start and (r, c) != maze.goal:
                display_grid[r, c] = 0.8  # Path tint

    # Colormap: 0=Open(white), 0.4=Explored(light cyan), 0.8=Path(orange), 1.0=Wall(charcoal)
    cmap = plt.cm.colors.ListedColormap(["#FFFFFF", "#B2EBF2", "#FFB74D", "#263238"])
    bounds = [-0.1, 0.2, 0.6, 0.9, 1.1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(display_grid, cmap=cmap, norm=norm, origin="upper")

    # Major and minor grid ticks
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#CFD8DC", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", size=0)

    # Start & Goal markers
    sr, sc = maze.start
    gr, gc = maze.goal
    ax.add_patch(
        patches.Rectangle(
            (sc - 0.45, sr - 0.45),
            0.9,
            0.9,
            facecolor="#4CAF50",
            edgecolor="#2E7D32",
            linewidth=2,
            zorder=4,
        )
    )
    ax.text(
        sc,
        sr,
        "S",
        color="white",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=12,
        zorder=5,
    )

    ax.add_patch(
        patches.Rectangle(
            (gc - 0.45, gr - 0.45),
            0.9,
            0.9,
            facecolor="#E53935",
            edgecolor="#B71C1C",
            linewidth=2,
            zorder=4,
        )
    )
    ax.text(
        gc,
        gr,
        "G",
        color="white",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=12,
        zorder=5,
    )

    # Path polyline overlay for clarity
    if show_path and len(result.path) > 1:
        pr = [p[0] for p in result.path]
        pc = [p[1] for p in result.path]
        ax.plot(pc, pr, color="#E65100", linewidth=2.5, linestyle="-", zorder=6)

    ax.set_xticks(range(0, cols, max(1, cols // 5)))
    ax.set_yticks(range(0, rows, max(1, rows // 5)))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


def plot_eight_puzzle(
    state: Tuple[int, ...],
    title: str = "8-Puzzle Board",
    figsize: Tuple[int, int] = (4, 4),
) -> plt.Figure:
    """Renders a clean 3x3 sliding tile board with tile numbers."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.invert_yaxis()
    ax.axis("off")

    for idx, val in enumerate(state):
        r, c = divmod(idx, 3)
        if val == 0:
            # Blank tile
            rect = patches.Rectangle(
                (c + 0.05, r + 0.05),
                0.9,
                0.9,
                facecolor="#ECEFF1",
                edgecolor="#B0BEC5",
                linewidth=2,
                linestyle="--",
            )
            ax.add_patch(rect)
        else:
            rect = patches.Rectangle(
                (c + 0.05, r + 0.05),
                0.9,
                0.9,
                facecolor="#1976D2",
                edgecolor="#0D47A1",
                linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(
                c + 0.5,
                r + 0.5,
                str(val),
                color="white",
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
            )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    plt.tight_layout()
    return fig


def plot_ga_convergence(
    history_df: pd.DataFrame,
    target_length: int,
    figsize: Tuple[int, int] = (8, 4),
) -> plt.Figure:
    """Plots Best and Average Fitness progression across generations."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_theme(style="whitegrid")

    ax.plot(
        history_df["generation"],
        history_df["best_fitness"],
        label="Best Fitness",
        color="#2E7D32",
        linewidth=2.5,
    )
    ax.plot(
        history_df["generation"],
        history_df["avg_fitness"],
        label="Average Fitness",
        color="#0288D1",
        linewidth=1.8,
        linestyle="--",
    )

    ax.set_title(
        f"Evolutionary Convergence Curve (Target Path = {target_length} Steps)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Generation", fontsize=10)
    ax.set_ylabel("Fitness Score", fontsize=10)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_confusion_matrix_heatmap(
    conf_matrix: List[List[int]],
    classes: List[str],
    accuracy: float,
    figsize: Tuple[int, int] = (5, 4),
) -> plt.Figure:
    """Plots Seaborn heatmap for Decision Tree confusion matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    cm_arr = np.array(conf_matrix)

    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar=False,
    )
    ax.set_title(f"Decision Tree Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Algorithm", fontsize=10)
    ax.set_ylabel("True Algorithm", fontsize=10)
    plt.tight_layout()
    return fig


def plot_decision_tree_graph(
    clf: Any,
    feature_names: List[str],
    class_names: List[str],
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Renders the decision tree structure with split rules."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax,
        max_depth=3,  # Top levels for visual clarity
    )
    ax.set_title("Decision Tree Logic (Top 3 Depths Shown for Readability)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    r2: float,
    rmse: float,
    figsize: Tuple[int, int] = (6, 5),
) -> plt.Figure:
    """Plots Linear Regression actual vs. predicted scatter with y=x line."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_theme(style="whitegrid")

    ax.scatter(y_actual, y_pred, alpha=0.5, color="#1565C0", edgecolors="none", s=25, label="Test Samples")

    # Ideal diagonal line
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="#D32F2F", linestyle="--", linewidth=2, label="Ideal Line (y = x)")

    ax.set_title(f"A* Path Cost: Actual vs. Predicted\n$R^2 = {r2:.4f}$ | RMSE = {rmse:.2f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual A* Path Cost", fontsize=10)
    ax.set_ylabel("Predicted Path Cost", fontsize=10)
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: List[str],
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plots correlation heatmap across benchmark dataset features."""
    fig, ax = plt.subplots(figsize=figsize)
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar=True)
    ax.set_title("Feature Correlation Heatmap", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig
