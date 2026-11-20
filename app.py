"""
MazeMind AI: Intelligent Puzzle Solver & Search Algorithm Visualizer.
Theme: Official Dracula Theme (https://draculatheme.com/spec)
Features:
- Pure Python Classical Search (BFS, DFS, Greedy Best-First, A*)
- 2D Grid Maze with step-by-step auto-animation or instant solution
- 8-Puzzle Sliding Tiles with 1-by-1 tile movement animation and Inversion Parity
- Compact responsive single-viewport layout (zero unnecessary scrolling)
"""

import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from mazemind.environments import GridMaze, EightPuzzle
from mazemind.search_agents import (
    SearchResult,
    BFSAgent,
    DFSAgent,
    GreedyAgent,
    AStarAgent,
)
from mazemind.visualizer import (
    plot_maze,
    plot_eight_puzzle,
    plot_comparison_bars,
    DRACULA,
)

# ---------------------------------------------------------
# Page Configuration & Dracula Global Styling (Fit to Screen)
# ---------------------------------------------------------
st.set_page_config(
    page_title="MazeMind AI — Search & Puzzle Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Dracula Theme Specification: https://draculatheme.com/spec */
    .stApp {
        background-color: #282a36;
        color: #f8f8f2;
    }
    [data-testid="stSidebar"] {
        background-color: #21222c;
        border-right: 1px solid #44475a;
    }
    /* Eliminate excessive top whitespace so all content fits on screen */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2rem !important;
    }
    .header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 6px;
        margin-bottom: 8px;
        border-bottom: 1px solid #44475a;
    }
    .header-title {
        font-size: 1.45rem;
        font-weight: 800;
        background: linear-gradient(90deg, #bd93f9, #ff79c6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-badge {
        font-size: 0.8rem;
        color: #8be9fd;
        background: #383a59;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 500;
    }
    .metric-pill {
        background-color: #21222c;
        border: 1px solid #44475a;
        border-left: 4px solid #bd93f9;
        border-radius: 6px;
        padding: 6px 10px;
        color: #f8f8f2;
        font-size: 0.82rem;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    .action-badge {
        background-color: #44475a;
        border: 1px solid #ff79c6;
        border-radius: 5px;
        padding: 3px 8px;
        font-weight: 600;
        color: #ff79c6;
        display: inline-block;
        margin-bottom: 4px;
        font-size: 0.82rem;
    }
    .stButton>button {
        background-color: #bd93f9;
        color: #282a36;
        font-weight: 700;
        border: none;
        border-radius: 6px;
        padding: 4px 10px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #ff79c6;
        color: #282a36;
    }
    /* Compact input controls */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.4rem !important;
    }
    .stSelectbox, .stRadio, .stSlider {
        margin-bottom: 2px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "grid_size" not in st.session_state:
    st.session_state.grid_size = 15

if "maze_grid" not in st.session_state:
    rng = np.random.default_rng(42)
    dim = st.session_state.grid_size
    grid = (rng.random((dim, dim)) < 0.22).astype(np.int32)
    grid[0, 0] = 0
    grid[dim - 1, dim - 1] = 0
    st.session_state.maze_grid = grid

if "search_results" not in st.session_state:
    st.session_state.search_results = {}

if "maze_anim_step" not in st.session_state:
    st.session_state.maze_anim_step = None

if "trigger_maze_anim" not in st.session_state:
    st.session_state.trigger_maze_anim = False

if "puzzle_state" not in st.session_state:
    st.session_state.puzzle_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)

if "puzzle_goal" not in st.session_state:
    st.session_state.puzzle_goal = EightPuzzle.STANDARD_GOAL

if "puzzle_results" not in st.session_state:
    st.session_state.puzzle_results = {}

if "puzzle_anim_step" not in st.session_state:
    st.session_state.puzzle_anim_step = 0

if "trigger_puzzle_anim" not in st.session_state:
    st.session_state.trigger_puzzle_anim = False

# ---------------------------------------------------------
# Sidebar: Navigation & Clean Styling
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 MazeMind AI")
    st.caption("Classical Search & Puzzle Visualizer")
    st.markdown("---")

    nav_selection = st.radio(
        "Select Problem / View:",
        [
            "1. 2D Grid Maze Solver",
            "2. 8-Puzzle State-Space Solver",
            "3. Algorithm Theory & Complexity",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.82rem; color:#6272a4;">
            🎨 Styled with official <b><a href="https://draculatheme.com/spec" target="_blank" style="color:#bd93f9; text-decoration:none;">Dracula Theme</a></b>.<br>
            ⚡ Pure Python Search (Zero External AI APIs).
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# VIEW 1: 2D Grid Maze Solver & Visualizer
# =========================================================
if nav_selection == "1. 2D Grid Maze Solver":
    st.markdown(
        """
        <div class="header-bar">
            <span class="header-title">2D Grid Maze Solver & Visualizer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_ctrl, col_view = st.columns([1, 1.25])

    with col_ctrl:
        st.markdown("##### 🗺️ Maze Layout Controls")

        c_size, c_dens = st.columns(2)
        with c_size:
            dim = st.slider(
                "Grid Size (N×N):",
                min_value=8,
                max_value=30,
                value=st.session_state.grid_size,
            )
        with c_dens:
            density = st.slider("Obstacle Density:", 0.05, 0.45, 0.22, 0.01)

        c_preset, c_gen_btn = st.columns([1.3, 1])
        with c_preset:
            preset = st.selectbox(
                "Preset Pattern:",
                [
                    "Random Obstacles",
                    "Open Room (No Walls)",
                    "Center Wall Barrier",
                    "Snake Labyrinth",
                ],
                label_visibility="collapsed",
            )
        with c_gen_btn:
            if st.button("🎲 New Maze", use_container_width=True):
                st.session_state.grid_size = dim
                st.session_state.search_results = {}
                st.session_state.maze_anim_step = None
                st.session_state.trigger_maze_anim = False

                if preset == "Open Room (No Walls)":
                    new_grid = np.zeros((dim, dim), dtype=np.int32)
                elif preset == "Center Wall Barrier":
                    new_grid = np.zeros((dim, dim), dtype=np.int32)
                    mid = dim // 2
                    new_grid[0 : dim - 2, mid] = 1
                elif preset == "Snake Labyrinth":
                    new_grid = np.zeros((dim, dim), dtype=np.int32)
                    for r in range(2, dim - 1, 3):
                        if (r // 3) % 2 == 0:
                            new_grid[r, 1:] = 1
                        else:
                            new_grid[r, :-1] = 1
                else:
                    rng = np.random.default_rng()
                    new_grid = (rng.random((dim, dim)) < density).astype(np.int32)

                new_grid[0, 0] = 0
                new_grid[dim - 1, dim - 1] = 0
                st.session_state.maze_grid = new_grid
                st.rerun()

        curr_maze = GridMaze(
            grid=st.session_state.maze_grid,
            start=(0, 0),
            goal=(
                st.session_state.maze_grid.shape[0] - 1,
                st.session_state.maze_grid.shape[1] - 1,
            ),
        )
        stats = curr_maze.get_stats()

        st.markdown(
            f"""
            <div class="metric-pill">
                📏 <b>{stats['rows']}×{stats['cols']}</b> ({stats['total_cells']} cells) &nbsp;|&nbsp;
                🧱 <b>{stats['wall_density']*100:.1f}% walls</b> ({stats['wall_cells']} walls) &nbsp;|&nbsp;
                🎯 Dist: <b>{stats['manhattan_dist']}</b> &nbsp;|&nbsp;
                🛑 Dead Ends: <b>{stats['dead_end_count']}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### ⚡ Solver Setup")

        c_alg, c_h = st.columns([1.3, 1])
        with c_alg:
            solver_selection = st.selectbox(
                "Search Strategy:",
                [
                    "A* Search",
                    "Breadth-First Search (BFS)",
                    "Depth-First Search (DFS)",
                    "Greedy Best-First Search",
                    "Run All 4 (Comparison)",
                ],
            )
        with c_h:
            heuristic_choice = st.selectbox(
                "Heuristic Metric:",
                ["Manhattan (L1)", "Euclidean (L2)"],
            )
            metric_key = "manhattan" if "Manhattan" in heuristic_choice else "euclidean"

        # Explicit option requested: Animated step-by-step vs Instant solution!
        solve_mode = st.radio(
            "Execution Mode:",
            ["🎬 Animated (Step-by-Step)", "⚡ Instant Solution (No Animation)"],
            horizontal=True,
        )

        if st.button("🚀 Solve Maze", type="primary", use_container_width=True):
            agents = []
            if solver_selection.startswith("Run All"):
                agents = [
                    AStarAgent(name=f"A* ({metric_key.title()})", heuristic_metric=metric_key),
                    BFSAgent(name="BFS"),
                    DFSAgent(name="DFS"),
                    GreedyAgent(name=f"Greedy ({metric_key.title()})", heuristic_metric=metric_key),
                ]
            else:
                agent_map = {
                    "A* Search": AStarAgent(name=f"A* ({metric_key.title()})", heuristic_metric=metric_key),
                    "Breadth-First Search (BFS)": BFSAgent(name="BFS"),
                    "Depth-First Search (DFS)": DFSAgent(name="DFS"),
                    "Greedy Best-First Search": GreedyAgent(name=f"Greedy ({metric_key.title()})", heuristic_metric=metric_key),
                }
                agents = [agent_map[solver_selection]]

            results = {}
            for ag in agents:
                results[ag.name] = ag.solve(curr_maze)
            st.session_state.search_results = results

            # If animated mode is selected, immediately trigger step-by-step animation!
            if "Animated" in solve_mode:
                st.session_state.trigger_maze_anim = True
                st.session_state.maze_anim_step = 1
            else:
                st.session_state.trigger_maze_anim = False
                st.session_state.maze_anim_step = None
            st.rerun()

    with col_view:
        if st.session_state.search_results:
            agent_names = list(st.session_state.search_results.keys())

            c_pick, c_spd = st.columns([1.4, 1.2])
            with c_pick:
                selected_agent_name = st.selectbox(
                    "Displaying Solver:",
                    agent_names,
                    index=0,
                    label_visibility="collapsed",
                )
            with c_spd:
                maze_speed_label = st.select_slider(
                    "Speed:",
                    options=["0.25x (Slow)", "0.5x", "1.0x (Normal)", "2.0x (Fast)", "5.0x (Turbo)"],
                    value="1.0x (Normal)",
                    key="slider_maze_speed",
                    label_visibility="collapsed",
                )

            active_res = st.session_state.search_results[selected_agent_name]

            if active_res.found:
                step_count = len(active_res.exploration_history)

                # Playback Controls Bar
                pb1, pb2, pb3, pb4 = st.columns([1, 1, 1, 2])
                with pb1:
                    replay_maze = st.button("▶️ Replay", use_container_width=True, key="btn_replay_maze")
                with pb2:
                    pause_maze = st.button("⏸️ Pause", use_container_width=True, key="btn_pause_maze")
                with pb3:
                    reset_maze = st.button("⏮️ Reset", use_container_width=True, key="btn_reset_maze")
                with pb4:
                    st.caption(
                        f"Path: **{active_res.path_cost}** steps | Explored: **{active_res.nodes_expanded}** | Time: **{active_res.execution_time_ms:.1f}ms**"
                    )

                if reset_maze:
                    st.session_state.maze_anim_step = 1
                    st.session_state.trigger_maze_anim = False

                plot_slot = st.empty()
                progress_slot = st.empty()

                speed_map = {
                    "0.25x (Slow)": (0.10, 1),
                    "0.5x": (0.04, 1),
                    "1.0x (Normal)": (0.015, 2),
                    "2.0x (Fast)": (0.008, 4),
                    "5.0x (Turbo)": (0.003, 8),
                }
                delay, stride = speed_map[maze_speed_label]

                # Auto-run animation when clicking Solve Maze (if animated mode), or when Replay is clicked
                run_animation = st.session_state.get("trigger_maze_anim", False) or replay_maze

                if run_animation and not pause_maze:
                    st.session_state.trigger_maze_anim = False
                    start_s = 1 if replay_maze else (st.session_state.maze_anim_step or 1)
                    if start_s >= step_count:
                        start_s = 1

                    for s in range(start_s, step_count + 1, stride):
                        st.session_state.maze_anim_step = s
                        fig = plot_maze(
                            curr_maze,
                            result=active_res,
                            step=s,
                            title=f"{selected_agent_name} Searching... ({s}/{step_count} nodes)",
                            figsize=(4.4, 4.4),
                        )
                        plot_slot.pyplot(fig)
                        plt.close(fig)
                        progress_slot.progress(
                            s / step_count, text=f"Exploring: {s}/{step_count}"
                        )
                        time.sleep(delay)

                    st.session_state.maze_anim_step = step_count
                    fig = plot_maze(
                        curr_maze,
                        result=active_res,
                        step=step_count,
                        title=f"✅ {selected_agent_name} Solved! (Optimal Path: {active_res.path_cost})",
                        figsize=(4.4, 4.4),
                    )
                    plot_slot.pyplot(fig)
                    plt.close(fig)
                    progress_slot.empty()
                else:
                    cur_val = st.session_state.maze_anim_step
                    if cur_val is None:
                        cur_val = step_count
                    cur_val = max(1, min(step_count, cur_val))

                    fig = plot_maze(
                        curr_maze,
                        result=active_res,
                        step=cur_val,
                        title=f"{selected_agent_name} (Step {cur_val}/{step_count} — Cost: {active_res.path_cost})",
                        figsize=(4.4, 4.4),
                    )
                    plot_slot.pyplot(fig)
                    plt.close(fig)
            else:
                st.error("❌ No path found between Start (0, 0) and Goal!")
                fig = plot_maze(
                    curr_maze,
                    result=active_res,
                    title=f"{selected_agent_name} (Unsolvable)",
                    figsize=(4.4, 4.4),
                )
                st.pyplot(fig)
                plt.close(fig)
        else:
            fig = plot_maze(
                curr_maze,
                title=f"Initial Grid ({curr_maze.rows}×{curr_maze.cols})",
                figsize=(4.4, 4.4),
            )
            st.pyplot(fig)
            plt.close(fig)

    # Performance Comparison in a Clean Expander Below (Preserves Single Screen Height)
    if st.session_state.search_results and len(st.session_state.search_results) > 1:
        with st.expander("📊 View Search Algorithm Comparison Table & Charts", expanded=False):
            rows_data = []
            for name, res in st.session_state.search_results.items():
                rows_data.append(
                    {
                        "Algorithm": name,
                        "Heuristic": res.heuristic_metric.title(),
                        "Status": "✅ Solved" if res.found else "❌ Failed",
                        "Path Length": res.path_cost if res.found else "N/A",
                        "Nodes Expanded": res.nodes_expanded,
                        "Max Frontier": res.max_frontier_size,
                        "Time (ms)": f"{res.execution_time_ms:.3f}",
                    }
                )
            st.dataframe(pd.DataFrame(rows_data), use_container_width=True, hide_index=True)
            fig_bars = plot_comparison_bars(st.session_state.search_results)
            st.pyplot(fig_bars)
            plt.close(fig_bars)


# =========================================================
# VIEW 2: 8-Puzzle State-Space Solver & Visualizer
# =========================================================
elif nav_selection == "2. 8-Puzzle State-Space Solver":
    st.markdown(
        """
        <div class="header-bar">
            <span class="header-title">8-Puzzle State-Space Solver & Visualizer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p_col1, p_col2 = st.columns([1, 1.25])

    with p_col1:
        st.markdown("##### ⚙️ Initial Board Setup")

        p_pre, p_goal = st.columns(2)
        with p_pre:
            preset_puzzle = st.selectbox(
                "Preset Configuration:",
                [
                    "1-Move Trivial Solution",
                    "3-Moves Easy",
                    "6-Moves Medium",
                    "10-Moves Challenging",
                    "Random Solvable Board",
                ],
            )
        with p_goal:
            goal_choice = st.selectbox(
                "Target Goal:",
                [
                    "Standard: (1..8, 0)",
                    "Spiral: (1,2,3,8,0,4,7,6,5)",
                ],
            )
        selected_goal = (
            EightPuzzle.STANDARD_GOAL
            if "Standard" in goal_choice
            else EightPuzzle.SPIRAL_GOAL
        )

        if st.button("🎲 Scramble / Apply Board", use_container_width=True):
            if preset_puzzle == "1-Move Trivial Solution":
                st.session_state.puzzle_state = (1, 2, 3, 4, 5, 6, 7, 0, 8)
            elif preset_puzzle == "3-Moves Easy":
                st.session_state.puzzle_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)
            elif preset_puzzle == "6-Moves Medium":
                st.session_state.puzzle_state = (1, 2, 3, 0, 4, 6, 7, 5, 8)
            elif preset_puzzle == "10-Moves Challenging":
                st.session_state.puzzle_state = (1, 3, 6, 4, 2, 5, 7, 0, 8)
            else:
                while True:
                    cand = list(range(9))
                    random.shuffle(cand)
                    cand_tuple = tuple(cand)
                    if EightPuzzle.is_solvable(cand_tuple, selected_goal):
                        st.session_state.puzzle_state = cand_tuple
                        break

            st.session_state.puzzle_goal = selected_goal
            st.session_state.puzzle_results = {}
            st.session_state.puzzle_anim_step = 0
            st.session_state.trigger_puzzle_anim = False
            st.rerun()

        inv_count = EightPuzzle.count_inversions(st.session_state.puzzle_state)
        goal_inv = EightPuzzle.count_inversions(st.session_state.puzzle_goal)
        is_solv = (inv_count % 2) == (goal_inv % 2)

        st.markdown(
            f"""
            <div class="metric-pill">
                🔍 <b>Inversions:</b> Start = <b>{inv_count}</b> ({'Even' if inv_count % 2 == 0 else 'Odd'}) &nbsp;|&nbsp;
                Goal = <b>{goal_inv}</b> &nbsp;|&nbsp;
                Status: <b>{'🟢 Mathematically Solvable' if is_solv else '🔴 Unsolvable'}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### ⚡ Solver Setup")
        p_c_strat, p_c_heur = st.columns([1.3, 1])
        with p_c_strat:
            p_solver_choice = st.selectbox(
                "Search Strategy:",
                [
                    "A* Search",
                    "Greedy Best-First",
                    "Breadth-First Search (BFS)",
                    "Depth-First Search (DFS)",
                    "Run All 4 (Comparison)",
                ],
            )
        with p_c_heur:
            p_metric = st.selectbox(
                "Heuristic:",
                ["Manhattan", "Euclidean", "Misplaced"],
            )
            metric_key = p_metric.lower()

        # Explicit option: Animated vs Instant!
        puzzle_solve_mode = st.radio(
            "Execution Mode:",
            ["🎬 Animated (Tiles Slide 1-by-1)", "⚡ Instant Solution (No Animation)"],
            horizontal=True,
            key="p_solve_mode_radio",
        )

        if st.button("🚀 Solve 8-Puzzle", type="primary", use_container_width=True):
            env_puzzle = EightPuzzle(
                start_state=st.session_state.puzzle_state,
                goal_state=st.session_state.puzzle_goal,
            )

            if p_solver_choice.startswith("Run All"):
                agents = [
                    AStarAgent(name=f"A* ({metric_key.title()})", heuristic_metric=metric_key),
                    GreedyAgent(name=f"Greedy ({metric_key.title()})", heuristic_metric=metric_key),
                    BFSAgent(name="BFS"),
                    DFSAgent(name="DFS"),
                ]
            else:
                agent_dict = {
                    "A* Search": AStarAgent(name=f"A* ({metric_key.title()})", heuristic_metric=metric_key),
                    "Greedy Best-First": GreedyAgent(name=f"Greedy ({metric_key.title()})", heuristic_metric=metric_key),
                    "Breadth-First Search (BFS)": BFSAgent(name="BFS"),
                    "Depth-First Search (DFS)": DFSAgent(name="DFS"),
                }
                agents = [agent_dict[p_solver_choice]]

            p_results = {}
            with st.spinner("Solving puzzle..."):
                for ag in agents:
                    p_results[ag.name] = ag.solve(env_puzzle)
            st.session_state.puzzle_results = p_results

            if "Animated" in puzzle_solve_mode:
                st.session_state.trigger_puzzle_anim = True
                st.session_state.puzzle_anim_step = 0
            else:
                st.session_state.trigger_puzzle_anim = False
                st.session_state.puzzle_anim_step = None
            st.rerun()

    with p_col2:
        if st.session_state.puzzle_results:
            p_names = list(st.session_state.puzzle_results.keys())
            c_p_sel, c_p_spd = st.columns([1.4, 1.2])
            with c_p_sel:
                active_p_name = st.selectbox(
                    "Displaying Solver:",
                    p_names,
                    index=0,
                    label_visibility="collapsed",
                )
            with c_p_spd:
                puzzle_speed_label = st.select_slider(
                    "Speed:",
                    options=["Slow (0.8s)", "Normal (0.35s)", "Fast (0.15s)", "Turbo (0.05s)"],
                    value="Normal (0.35s)",
                    key="slider_puzzle_speed",
                    label_visibility="collapsed",
                )

            res_p = st.session_state.puzzle_results[active_p_name]

            if res_p and res_p.found:
                # Precompute tile movements
                path_moves: List[Tuple[Optional[int], str]] = [(None, "Initial Board State")]
                for idx in range(1, len(res_p.path)):
                    prev_b = res_p.path[idx - 1]
                    curr_b = res_p.path[idx]
                    prev_zero = prev_b.index(0)
                    curr_zero = curr_b.index(0)
                    moved_tile = prev_b[curr_zero]

                    r_prev, c_prev = divmod(curr_zero, 3)
                    r_curr, c_curr = divmod(prev_zero, 3)
                    dr, dc = r_curr - r_prev, c_curr - c_prev
                    direction = "UP" if dr == -1 else ("DOWN" if dr == 1 else ("LEFT" if dc == -1 else "RIGHT"))
                    path_moves.append((moved_tile, f"Step {idx}: Tile {moved_tile} slid {direction} into blank slot"))

                # Playback controls bar
                pc1, pc2, pc3, pc4 = st.columns([1, 1, 1, 2])
                with pc1:
                    replay_puzzle = st.button("▶️ Replay", use_container_width=True, key="btn_replay_puzzle")
                with pc2:
                    pause_puzzle = st.button("⏸️ Pause", use_container_width=True, key="btn_pause_puzzle")
                with pc3:
                    reset_puzzle = st.button("⏮️ Reset", use_container_width=True, key="btn_reset_puzzle")
                with pc4:
                    st.caption(
                        f"Moves: **{res_p.path_cost}** | Nodes: **{res_p.nodes_expanded}** | Time: **{res_p.execution_time_ms:.1f}ms**"
                    )

                if reset_puzzle:
                    st.session_state.puzzle_anim_step = 0
                    st.session_state.trigger_puzzle_anim = False

                p_slot = st.empty()
                info_slot = st.empty()

                p_speed_map = {
                    "Slow (0.8s)": 0.8,
                    "Normal (0.35s)": 0.35,
                    "Fast (0.15s)": 0.15,
                    "Turbo (0.05s)": 0.05,
                }
                p_delay = p_speed_map[puzzle_speed_label]

                run_p_anim = st.session_state.get("trigger_puzzle_anim", False) or replay_puzzle

                if run_p_anim and not pause_puzzle:
                    st.session_state.trigger_puzzle_anim = False
                    start_step = 0 if replay_puzzle else (st.session_state.puzzle_anim_step or 0)
                    if start_step >= len(res_p.path) - 1:
                        start_step = 0

                    for s in range(start_step, len(res_p.path)):
                        st.session_state.puzzle_anim_step = s
                        step_board = res_p.path[s]
                        moved_t, desc = path_moves[s]

                        info_slot.markdown(
                            f'<div class="action-badge">📍 {desc}</div>',
                            unsafe_allow_html=True,
                        )
                        fig = plot_eight_puzzle(
                            step_board,
                            title=f"Step {s}/{len(res_p.path) - 1} ({'Goal Reached!' if s == len(res_p.path) - 1 else 'Sliding...'})",
                            moved_tile=moved_t,
                            move_action=desc,
                            figsize=(3.8, 3.8),
                        )
                        p_slot.pyplot(fig)
                        plt.close(fig)
                        time.sleep(p_delay)

                    st.session_state.puzzle_anim_step = len(res_p.path) - 1
                else:
                    curr_step = st.session_state.puzzle_anim_step
                    if curr_step is None:
                        curr_step = len(res_p.path) - 1
                    curr_step = max(0, min(len(res_p.path) - 1, curr_step))

                    curr_board = res_p.path[curr_step]
                    moved_t, desc = path_moves[curr_step]

                    info_slot.markdown(
                        f'<div class="action-badge">📍 {desc}</div>',
                        unsafe_allow_html=True,
                    )
                    fig = plot_eight_puzzle(
                        curr_board,
                        title=f"Step {curr_step}/{len(res_p.path) - 1} ({'Goal Reached!' if curr_step == len(res_p.path) - 1 else 'State Preview'})",
                        moved_tile=moved_t,
                        move_action=desc,
                        figsize=(3.8, 3.8),
                    )
                    p_slot.pyplot(fig)
                    plt.close(fig)

                with st.expander("📜 View Step-by-Step Move Log", expanded=False):
                    move_table = [
                        {"Step": idx, "Action": path_moves[idx][1]}
                        for idx in range(1, len(path_moves))
                    ]
                    st.dataframe(pd.DataFrame(move_table), use_container_width=True, hide_index=True)
            else:
                st.error("No path found to goal.")
                fig = plot_eight_puzzle(
                    st.session_state.puzzle_state,
                    title="Initial State (Unsolvable)",
                    figsize=(3.8, 3.8),
                )
                st.pyplot(fig)
                plt.close(fig)
        else:
            fig = plot_eight_puzzle(
                st.session_state.puzzle_state,
                title="Current Initial State",
                figsize=(3.8, 3.8),
            )
            st.pyplot(fig)
            plt.close(fig)

    if st.session_state.puzzle_results and len(st.session_state.puzzle_results) > 1:
        with st.expander("📊 View 8-Puzzle Multi-Algorithm Comparison Table", expanded=False):
            p_table_rows = []
            for name, res in st.session_state.puzzle_results.items():
                p_table_rows.append(
                    {
                        "Algorithm": name,
                        "Status": "✅ Solved" if res.found else "❌ Failed",
                        "Path Length": res.path_cost if res.found else "N/A",
                        "Nodes Expanded": res.nodes_expanded,
                        "Max Frontier": res.max_frontier_size,
                        "Time (ms)": f"{res.execution_time_ms:.3f}",
                    }
                )
            st.dataframe(pd.DataFrame(p_table_rows), use_container_width=True, hide_index=True)


# =========================================================
# VIEW 3: Algorithm Theory & Complexity
# =========================================================
elif nav_selection == "3. Algorithm Theory & Complexity":
    st.markdown(
        """
        <div class="header-bar">
            <span class="header-title">Search Algorithm Theory & Complexity</span>
            <span class="header-badge">Mathematical Proofs & Space-Time Analysis</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### 1. Comparative Complexity Matrix")
    theory_data = [
        {
            "Algorithm": "Breadth-First Search (BFS)",
            "Type": "Uninformed (Blind)",
            "Evaluation": "g(n) = depth(n)",
            "Data Structure": "FIFO Queue",
            "Time": "O(b^d)",
            "Space": "O(b^d)",
            "Optimal?": "Yes (unit cost)",
        },
        {
            "Algorithm": "Depth-First Search (DFS)",
            "Type": "Uninformed (Blind)",
            "Evaluation": "LIFO order",
            "Data Structure": "LIFO Stack",
            "Time": "O(b^m)",
            "Space": "O(b · m)",
            "Optimal?": "No",
        },
        {
            "Algorithm": "Greedy Best-First Search",
            "Type": "Informed (Heuristic)",
            "Evaluation": "f(n) = h(n)",
            "Data Structure": "Priority Queue",
            "Time": "O(b^m)",
            "Space": "O(b^m)",
            "Optimal?": "No",
        },
        {
            "Algorithm": "A* Search",
            "Type": "Informed (Optimal)",
            "Evaluation": "f(n) = g(n) + h(n)",
            "Data Structure": "Priority Queue",
            "Time": "O(b^d) to O(b^m)",
            "Space": "O(b^d)",
            "Optimal?": "Yes (admissible h)",
        },
    ]
    st.dataframe(pd.DataFrame(theory_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("##### 2. Admissible Heuristics & Dominance")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown(
            r"""
            **Manhattan Distance ($L_1$ Norm):**
            $$h_{\text{Manhattan}}(n) = |r_n - r_{\text{goal}}| + |c_n - c_{\text{goal}}|$$
            * Admissible ($h(n) \le h^*(n)$) and Consistent on 4-way orthogonal grids.
            * Dominates Euclidean distance and expands fewer or equal nodes.
            """
        )
    with col_h2:
        st.markdown(
            r"""
            **Euclidean Distance ($L_2$ Norm):**
            $$h_{\text{Euclidean}}(n) = \sqrt{(r_n - r_{\text{goal}})^2 + (c_n - c_{\text{goal}})^2}$$
            * Admissible geometric shortest straight line.
            * Because $h_{\text{Manhattan}} \ge h_{\text{Euclidean}}$ on grids, Manhattan provides a tighter bound.
            """
        )

    st.markdown("---")
    st.markdown("##### 3. 8-Puzzle Solvability (Inversion Parity Theorem)")
    st.markdown(
        r"""
        $$\mathrm{Inv}(S) = \sum_{1 \le i < j \le 8} [S_i > S_j \land S_i \ne 0 \land S_j \ne 0]$$
        * Every horizontal slide has $\Delta \mathrm{Inv} = 0$.
        * Every vertical slide has $\Delta \mathrm{Inv} \in \{-2, 0, +2\}$.
        * Therefore, $\mathrm{Inv}(S) \equiv \mathrm{Inv}(G) \pmod 2$ is an invariant that mathematically guarantees solvability.
        """
    )
