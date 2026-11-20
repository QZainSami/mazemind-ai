"""
SearchVisualizer for MazeMind AI.
Builds Matplotlib figures for:
- 2D Grid Maze exploration states, frontiers, and optimal paths
- 8-Puzzle tile board states and transitions
- Multi-metric algorithm comparison bar charts (Nodes Expanded, Path Cost, Execution Time)
Course: CSL-325 Artificial Intelligence Lab (Fall 2026).
"""

from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Safe for headless Linux, Docker, and Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .environments import GridMaze, EightPuzzle
from .search_agents import SearchResult


def plot_maze(
    maze: GridMaze,
    result: Optional[SearchResult] = None,
    step: Optional[int] = None,
    title: str = "2D Grid Maze",
    figsize: Tuple[int, int] = (7, 7),
) -> plt.Figure:
    """
    Renders a 2D GridMaze with optional exploration overlay and final path.
    - If `step` is provided, shows exploration up to that step.
    """
    fig, ax = plt.subplots(figsize=figsize)
    rows, cols = maze.rows, maze.cols

    # Display grid: 0 = Open, 1 = Wall, 0.4 = Explored, 0.8 = Path
    display_grid = np.zeros((rows, cols), dtype=np.float32)
    display_grid[maze.grid == 1] = 1.0

    if result and result.exploration_history:
        history_slice = (
            result.exploration_history[:step]
            if step is not None
            else result.exploration_history
        )
        for r, c in history_slice:
            if (r, c) != maze.start and (r, c) != maze.goal and maze.grid[r, c] == 0:
                display_grid[r, c] = 0.4  # Explored tint

    show_path = (
        result
        and result.found
        and (step is None or step >= len(result.exploration_history))
    )
    if show_path and result.path:
        for r, c in result.path:
            if (r, c) != maze.start and (r, c) != maze.goal:
                display_grid[r, c] = 0.8  # Solution path tint

    # Colormap: Open (white), Explored (cyan), Path (amber/gold), Wall (charcoal)
    cmap = plt.cm.colors.ListedColormap(["#FFFFFF", "#B2EBF2", "#FFB74D", "#263238"])
    bounds = [-0.1, 0.2, 0.6, 0.9, 1.1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(display_grid, cmap=cmap, norm=norm, origin="upper")

    # Grid boundaries
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#CFD8DC", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", size=0)

    # Start marker (Green S)
    sr, sc = maze.start
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

    # Goal marker (Red G)
    gr, gc = maze.goal
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

    # Path polyline trace for extra visual clarity
    if show_path and len(result.path) > 1:
        pr = [p[0] for p in result.path]
        pc = [p[1] for p in result.path]
        ax.plot(pc, pr, color="#E65100", linewidth=2.5, linestyle="-", zorder=6)

    ax.set_xticks(range(0, cols, max(1, cols // 5)))
    ax.set_yticks(range(0, rows, max(1, rows // 5)))
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


def plot_eight_puzzle(
    state: Tuple[int, ...],
    title: str = "8-Puzzle Sliding Tiles",
    figsize: Tuple[int, int] = (4.5, 4.5),
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


def plot_comparison_bars(
    results: Dict[str, SearchResult],
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Renders side-by-side comparison bar charts displaying:
    - Nodes Expanded
    - Path Cost (Steps)
    - Execution Time (ms)
    """
    names = list(results.keys())
    nodes = [results[n].nodes_expanded for n in names]
    costs = [results[n].path_cost if results[n].found else 0 for n in names]
    times = [results[n].execution_time_ms for n in names]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    colors = ["#42A5F5", "#AB47BC", "#FFA726", "#66BB6A"]
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]

    # 1. Nodes Expanded
    bars1 = ax1.bar(names, nodes, color=bar_colors)
    ax1.set_title("Nodes Expanded (Lower is Better)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=25)
    for b in bars1:
        yval = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, yval, str(yval), ha="center", va="bottom", fontsize=8)

    # 2. Path Cost
    bars2 = ax2.bar(names, costs, color=bar_colors)
    ax2.set_title("Path Cost / Steps (Optimality)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Steps")
    ax2.tick_params(axis="x", rotation=25)
    for b in bars2:
        yval = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, yval, str(yval), ha="center", va="bottom", fontsize=8)

    # 3. Execution Time
    bars3 = ax3.bar(names, times, color=bar_colors)
    ax3.set_title("Execution Time (ms)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Milliseconds")
    ax3.tick_params(axis="x", rotation=25)
    for b in bars3:
        yval = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2, yval, f"{yval:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return fig
