"""
SearchVisualizer for MazeMind AI with Dracula Theme (https://draculatheme.com/spec).
Builds Matplotlib figures for:
- 2D Grid Maze exploration states, frontiers, and optimal paths
- 8-Puzzle tile board states, animations, and tile-move highlights
- Multi-metric algorithm comparison bar charts
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

# Official Dracula Theme Palette (https://draculatheme.com/spec)
DRACULA = {
    "bg": "#282a36",
    "current_line": "#44475a",
    "selection": "#44475a",
    "fg": "#f8f8f2",
    "comment": "#6272a4",
    "cyan": "#8be9fd",
    "green": "#50fa7b",
    "orange": "#ffb86c",
    "pink": "#ff79c6",
    "purple": "#bd93f9",
    "red": "#ff5555",
    "yellow": "#f1fa8c",
    "dark_wall": "#191a21",
}


def plot_maze(
    maze: GridMaze,
    result: Optional[SearchResult] = None,
    step: Optional[int] = None,
    title: str = "2D Grid Maze",
    figsize: Tuple[float, float] = (4.4, 4.4),
) -> plt.Figure:
    """
    Renders a 2D GridMaze in Dracula theme with exploration overlay and path.
    - If `step` is provided, shows exploration up to that step.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=DRACULA["bg"])
    ax.set_facecolor(DRACULA["bg"])
    rows, cols = maze.rows, maze.cols

    # Display grid:
    # 0 = Open, 0.2 = Mud(4), 0.3 = Water(8), 0.45 = Explored, 0.8 = Path, 1.0 = Wall
    display_grid = np.zeros((rows, cols), dtype=np.float32)
    if hasattr(maze, "weights"):
        display_grid[maze.weights >= 4] = 0.2
        display_grid[maze.weights >= 8] = 0.3
    display_grid[maze.grid == 1] = 1.0

    if result and result.exploration_history:
        history_slice = (
            result.exploration_history[:step]
            if step is not None
            else result.exploration_history
        )
        for r, c in history_slice:
            if (r, c) != maze.start and (r, c) != maze.goal and maze.grid[r, c] == 0:
                display_grid[r, c] = 0.45  # Explored tint

    show_path = (
        result
        and result.found
        and (step is None or step >= len(result.exploration_history))
    )
    if show_path and result.path:
        for r, c in result.path:
            if (r, c) != maze.start and (r, c) != maze.goal:
                display_grid[r, c] = 0.8  # Solution path tint

    cmap = plt.cm.colors.ListedColormap(
        [
            DRACULA["current_line"],
            "#6e473b",
            "#1e4d6b",
            DRACULA["purple"],
            DRACULA["yellow"],
            DRACULA["dark_wall"],
        ]
    )
    bounds = [-0.1, 0.1, 0.25, 0.35, 0.6, 0.9, 1.1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(display_grid, cmap=cmap, norm=norm, origin="upper")

    # Grid boundaries
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color=DRACULA["comment"], linestyle="-", linewidth=0.6, alpha=0.5)
    ax.tick_params(which="minor", size=0)

    # Start marker (Dracula Green)
    sr, sc = maze.start
    ax.add_patch(
        patches.Rectangle(
            (sc - 0.45, sr - 0.45),
            0.9,
            0.9,
            facecolor=DRACULA["green"],
            edgecolor=DRACULA["fg"],
            linewidth=2,
            zorder=4,
        )
    )
    ax.text(
        sc,
        sr,
        "S",
        color=DRACULA["bg"],
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=12,
        zorder=5,
    )

    # Goal marker (Dracula Red)
    gr, gc = maze.goal
    ax.add_patch(
        patches.Rectangle(
            (gc - 0.45, gr - 0.45),
            0.9,
            0.9,
            facecolor=DRACULA["red"],
            edgecolor=DRACULA["fg"],
            linewidth=2,
            zorder=4,
        )
    )
    ax.text(
        gc,
        gr,
        "G",
        color=DRACULA["bg"],
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=12,
        zorder=5,
    )

    # Path polyline trace in Dracula Orange
    if show_path and len(result.path) > 1:
        pr = [p[0] for p in result.path]
        pc = [p[1] for p in result.path]
        ax.plot(pc, pr, color=DRACULA["orange"], linewidth=2.8, linestyle="-", zorder=6)

    ax.set_xticks(range(0, cols, max(1, cols // 5)))
    ax.set_yticks(range(0, rows, max(1, rows // 5)))
    ax.tick_params(colors=DRACULA["fg"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(DRACULA["comment"])

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6, color=DRACULA["cyan"])
    plt.tight_layout(pad=0.4)
    return fig


def plot_eight_puzzle(
    state: Tuple[int, ...],
    title: str = "8-Puzzle Sliding Tiles",
    moved_tile: Optional[int] = None,
    move_action: Optional[str] = None,
    figsize: Tuple[float, float] = (3.8, 3.8),
) -> plt.Figure:
    """
    Renders a 3x3 sliding tile board in Dracula theme.
    - Highlights `moved_tile` in Dracula Pink/Cyan so the user can easily see which tile moved.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=DRACULA["bg"])
    ax.set_facecolor(DRACULA["bg"])
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.invert_yaxis()
    ax.axis("off")

    for idx, val in enumerate(state):
        r, c = divmod(idx, 3)
        if val == 0:
            # Blank tile slot
            rect = patches.Rectangle(
                (c + 0.06, r + 0.06),
                0.88,
                0.88,
                facecolor=DRACULA["dark_wall"],
                edgecolor=DRACULA["comment"],
                linewidth=2,
                linestyle="--",
            )
            ax.add_patch(rect)
        else:
            # Determine tile styling
            is_moved = (moved_tile is not None and val == moved_tile)
            face_color = DRACULA["pink"] if is_moved else DRACULA["current_line"]
            edge_color = DRACULA["yellow"] if is_moved else DRACULA["purple"]
            text_color = DRACULA["bg"] if is_moved else DRACULA["cyan"]
            line_width = 3.0 if is_moved else 2.0

            rect = patches.Rectangle(
                (c + 0.06, r + 0.06),
                0.88,
                0.88,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rect)
            ax.text(
                c + 0.5,
                r + 0.5,
                str(val),
                color=text_color,
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
            )

    full_title = title
    if move_action:
        full_title += f"\n{move_action}"

    ax.set_title(full_title, fontsize=10, fontweight="bold", pad=6, color=DRACULA["green"])
    plt.tight_layout(pad=0.4)
    return fig


def plot_comparison_bars(
    results: Dict[str, SearchResult],
    figsize: Tuple[int, int] = (10, 3.8),
) -> plt.Figure:
    """
    Renders side-by-side comparison bar charts in Dracula theme:
    - Nodes Expanded
    - Path Cost (Steps)
    - Execution Time (ms)
    """
    names = list(results.keys())
    nodes = [results[n].nodes_expanded for n in names]
    costs = [results[n].path_cost if results[n].found else 0 for n in names]
    times = [results[n].execution_time_ms for n in names]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, facecolor=DRACULA["bg"])

    palette = [DRACULA["cyan"], DRACULA["green"], DRACULA["orange"], DRACULA["purple"], DRACULA["pink"]]
    bar_colors = [palette[i % len(palette)] for i in range(len(names))]

    for ax, data, title, ylabel in [
        (ax1, nodes, "Nodes Expanded\n(Lower is Better)", "Count"),
        (ax2, costs, "Path Cost / Moves\n(Optimality)", "Steps / Moves"),
        (ax3, times, "Execution Time\n(Efficiency)", "ms"),
    ]:
        ax.set_facecolor(DRACULA["current_line"])
        bars = ax.bar(names, data, color=bar_colors, edgecolor=DRACULA["fg"], linewidth=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold", color=DRACULA["cyan"])
        ax.set_ylabel(ylabel, color=DRACULA["fg"], fontsize=9)
        ax.tick_params(axis="x", rotation=25, colors=DRACULA["fg"], labelsize=8)
        ax.tick_params(axis="y", colors=DRACULA["fg"], labelsize=8)
        ax.grid(axis="y", color=DRACULA["comment"], linestyle="--", alpha=0.3)
        for spine in ax.spines.values():
            spine.set_color(DRACULA["comment"])

        for b in bars:
            yval = b.get_height()
            label_text = f"{yval:.1f}" if isinstance(yval, float) else str(yval)
            ax.text(
                b.get_x() + b.get_width() / 2,
                yval,
                label_text,
                ha="center",
                va="bottom",
                fontsize=8,
                color=DRACULA["fg"],
                fontweight="bold",
            )

    plt.tight_layout()
    return fig
