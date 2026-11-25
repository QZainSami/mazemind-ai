"""
MazeMind AI: Intelligent Puzzle Solver & Search Algorithm Visualizer.
Theme: Official Dracula Theme (https://draculatheme.com/spec)
Features:
- Pure Python Classical Search (BFS, DFS, Greedy Best-First, A*)
- Guaranteed-Solvable 2D Grid Mazes across all grid sizes and obstacle densities
- Smooth 60 FPS Client-Side Tile-by-Tile Animation (Zero Re-Solve on Replay, Live Speed Slider)
- 8-Puzzle Sliding Tiles with 1-by-1 smooth tile sliding and Inversion Parity check
- Compact Single-Viewport Responsive Layout
"""

import os
import random
import sys
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from mazemind.environments import GridMaze, EightPuzzle
from mazemind.search_agents import (
    BFSAgent,
    DFSAgent,
    GreedyAgent,
    AStarAgent,
)
from mazemind.visualizer import plot_comparison_bars
from mazemind.web_animator import (
    build_maze_animator_html,
    build_puzzle_animator_html,
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
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 0.4rem !important;
        padding-left: 1.6rem !important;
        padding-right: 1.6rem !important;
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
        padding-bottom: 5px;
        margin-bottom: 6px;
        border-bottom: 1px solid #44475a;
    }
    .header-title {
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #bd93f9, #ff79c6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-pill {
        background-color: #21222c;
        border: 1px solid #44475a;
        border-left: 4px solid #bd93f9;
        border-radius: 6px;
        padding: 6px 10px;
        color: #f8f8f2;
        font-size: 0.82rem;
        margin-bottom: 6px;
        line-height: 1.35;
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
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.35rem !important;
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

if "grid_density" not in st.session_state:
    st.session_state.grid_density = 0.22

if "grid_preset" not in st.session_state:
    st.session_state.grid_preset = "Random Obstacles"

if "maze_grid" not in st.session_state:
    initial_maze = GridMaze.generate_solvable(
        rows=15,
        cols=15,
        density=0.22,
        preset="Random Obstacles",
        seed=42,
    )
    st.session_state.maze_grid = initial_maze.grid

if "search_results" not in st.session_state:
    st.session_state.search_results = {}

if "trigger_maze_anim" not in st.session_state:
    st.session_state.trigger_maze_anim = False

if "maze_instant_mode" not in st.session_state:
    st.session_state.maze_instant_mode = False

if "maze_solve_seq" not in st.session_state:
    st.session_state.maze_solve_seq = 0

if "puzzle_preset" not in st.session_state:
    st.session_state.puzzle_preset = "Random Solvable Board"

if "puzzle_goal_label" not in st.session_state:
    st.session_state.puzzle_goal_label = "Standard: (1..8, 0)"

if "puzzle_goal" not in st.session_state:
    st.session_state.puzzle_goal = EightPuzzle.STANDARD_GOAL

if "puzzle_state" not in st.session_state:
    st.session_state.puzzle_state = EightPuzzle.generate_random_solvable(
        goal_state=st.session_state.puzzle_goal,
        scramble_moves=18,
        seed=7,
    )

if "puzzle_results" not in st.session_state:
    st.session_state.puzzle_results = {}

if "trigger_puzzle_anim" not in st.session_state:
    st.session_state.trigger_puzzle_anim = False

if "puzzle_instant_mode" not in st.session_state:
    st.session_state.puzzle_instant_mode = False

if "puzzle_solve_seq" not in st.session_state:
    st.session_state.puzzle_solve_seq = 0

# ---------------------------------------------------------
# Sidebar Navigation
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
            🎨 Styled with official <b><a href="https://draculatheme.com/spec" target="_blank" style="color:#bd93f9; text-decoration:none;">Dracula Theme</a></b>.
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

    col_ctrl, col_view = st.columns([0.95, 1.35])

    with col_ctrl:
        st.markdown("##### 🗺️ Maze Layout Controls")

        c_size, c_dens = st.columns(2)
        with c_size:
            dim = st.slider(
                "Grid Size (N×N):",
                min_value=8,
                max_value=35,
                value=st.session_state.grid_size,
            )
        with c_dens:
            density = st.slider(
                "Obstacle Density:",
                min_value=0.05,
                max_value=0.50,
                value=st.session_state.grid_density,
                step=0.01,
            )

        preset_options = [
            "Random Obstacles",
            "Open Room (No Walls)",
            "Center Wall Barrier",
            "Snake Labyrinth",
        ]
        c_preset, c_gen_btn = st.columns([1.3, 1])
        with c_preset:
            preset = st.selectbox(
                "Preset Pattern:",
                preset_options,
                index=preset_options.index(st.session_state.grid_preset)
                if st.session_state.grid_preset in preset_options
                else 0,
                label_visibility="collapsed",
            )

        # Auto-regenerate guaranteed-solvable maze whenever sliders/preset change
        params_changed = (
            dim != st.session_state.grid_size
            or abs(density - st.session_state.grid_density) > 1e-9
            or preset != st.session_state.grid_preset
        )

        with c_gen_btn:
            new_maze_clicked = st.button("🎲 New Maze", width="stretch")

        if params_changed or new_maze_clicked:
            st.session_state.grid_size = dim
            st.session_state.grid_density = density
            st.session_state.grid_preset = preset
            st.session_state.search_results = {}
            st.session_state.trigger_maze_anim = False
            st.session_state.maze_instant_mode = False
            gen = GridMaze.generate_solvable(
                rows=dim,
                cols=dim,
                density=density,
                preset=preset,
            )
            st.session_state.maze_grid = gen.grid
            if new_maze_clicked:
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
                🧱 <b>{stats['wall_density']*100:.1f}% walls</b> ({stats['wall_cells']}) &nbsp;|&nbsp;
                🎯 Dist: <b>{stats['manhattan_dist']:.0f}</b> &nbsp;|&nbsp;
                ✅ <b>Guaranteed Solvable</b>
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
            is_uninformed_maze = solver_selection in (
                "Breadth-First Search (BFS)",
                "Depth-First Search (DFS)",
            )
            heuristic_choice = st.selectbox(
                "Heuristic Metric:",
                ["Manhattan (L1)", "Euclidean (L2)"],
                disabled=is_uninformed_maze,
                help="Used by informed solvers (A* and Greedy Best-First).",
            )
            metric_key = "manhattan" if "Manhattan" in heuristic_choice else "euclidean"

        solve_mode = st.radio(
            "Execution Mode:",
            ["🎬 Animated (Step-by-Step)", "⚡ Instant Solution (No Animation)"],
            horizontal=True,
        )

        if st.button("🚀 Solve Maze", type="primary", width="stretch"):
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
            st.session_state.maze_solve_seq += 1

            if "Animated" in solve_mode:
                st.session_state.trigger_maze_anim = True
                st.session_state.maze_instant_mode = False
            else:
                st.session_state.trigger_maze_anim = False
                st.session_state.maze_instant_mode = True
            st.rerun()

    with col_view:
        # Read autoplay once per solve trigger, then render completed on any later outer rerun
        autoplay_now = bool(st.session_state.trigger_maze_anim)
        if autoplay_now:
            st.session_state.trigger_maze_anim = False
            instant_now = False
        else:
            instant_now = bool(st.session_state.search_results) or st.session_state.maze_instant_mode

        maze_html = build_maze_animator_html(
            maze=curr_maze,
            results=st.session_state.search_results,
            autoplay=autoplay_now,
            instant=instant_now,
            canvas_px=360,
            run_id=st.session_state.maze_solve_seq,
        )
        st.iframe(maze_html, width="stretch", height=535)

    # Multi-Algorithm Benchmarking Expander below (fits main screen without scrolling)
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
            st.dataframe(pd.DataFrame(rows_data), width="stretch", hide_index=True)
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

    p_col1, p_col2 = st.columns([0.95, 1.35])

    with p_col1:
        st.markdown("##### ⚙️ Initial Board Setup")

        puzzle_presets = [
            "Random Solvable Board",
            "1-Move Trivial Solution",
            "3-Moves Easy",
            "6-Moves Medium",
            "10-Moves Challenging",
        ]
        goal_options = [
            "Standard: (1..8, 0)",
            "Spiral: (1,2,3,8,0,4,7,6,5)",
        ]

        p_pre, p_goal = st.columns(2)
        with p_pre:
            preset_puzzle = st.selectbox(
                "Preset Configuration:",
                puzzle_presets,
                index=puzzle_presets.index(st.session_state.puzzle_preset)
                if st.session_state.puzzle_preset in puzzle_presets
                else 0,
            )
        with p_goal:
            goal_choice = st.selectbox(
                "Target Goal:",
                goal_options,
                index=goal_options.index(st.session_state.puzzle_goal_label)
                if st.session_state.puzzle_goal_label in goal_options
                else 0,
            )
        selected_goal = (
            EightPuzzle.STANDARD_GOAL
            if "Standard" in goal_choice
            else EightPuzzle.SPIRAL_GOAL
        )

        p_params_changed = (
            preset_puzzle != st.session_state.puzzle_preset
            or goal_choice != st.session_state.puzzle_goal_label
        )
        scramble_clicked = st.button("🎲 Scramble / Apply Board", width="stretch")

        if p_params_changed or scramble_clicked:
            st.session_state.puzzle_preset = preset_puzzle
            st.session_state.puzzle_goal_label = goal_choice
            st.session_state.puzzle_goal = selected_goal

            if preset_puzzle == "Random Solvable Board":
                st.session_state.puzzle_state = EightPuzzle.generate_random_solvable(
                    goal_state=selected_goal, scramble_moves=18
                )
            elif "Standard" in goal_choice:
                fixed_presets = {
                    "1-Move Trivial Solution": (1, 2, 3, 4, 5, 6, 7, 0, 8),
                    "3-Moves Easy": (1, 2, 3, 4, 0, 6, 7, 5, 8),
                    "6-Moves Medium": (1, 2, 3, 0, 4, 6, 7, 5, 8),
                    "10-Moves Challenging": (1, 3, 6, 4, 2, 5, 7, 0, 8),
                }
                st.session_state.puzzle_state = fixed_presets[preset_puzzle]
            else:
                depth_map = {
                    "1-Move Trivial Solution": 1,
                    "3-Moves Easy": 3,
                    "6-Moves Medium": 6,
                    "10-Moves Challenging": 10,
                }
                st.session_state.puzzle_state = EightPuzzle.generate_random_solvable(
                    goal_state=selected_goal,
                    scramble_moves=depth_map[preset_puzzle],
                )

            st.session_state.puzzle_results = {}
            st.session_state.trigger_puzzle_anim = False
            st.session_state.puzzle_instant_mode = False
            if scramble_clicked:
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
            is_uninformed_p = p_solver_choice in (
                "Breadth-First Search (BFS)",
                "Depth-First Search (DFS)",
            )
            p_metric = st.selectbox(
                "Heuristic:",
                ["Manhattan", "Euclidean", "Misplaced"],
                disabled=is_uninformed_p,
                help="Used by informed solvers (A* and Greedy Best-First).",
            )
            metric_key = p_metric.lower()

        puzzle_solve_mode = st.radio(
            "Execution Mode:",
            ["🎬 Animated (Tiles Slide 1-by-1)", "⚡ Instant Solution (No Animation)"],
            horizontal=True,
            key="p_solve_mode_radio",
        )

        if st.button("🚀 Solve 8-Puzzle", type="primary", width="stretch"):
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
            st.session_state.puzzle_solve_seq += 1

            if "Animated" in puzzle_solve_mode:
                st.session_state.trigger_puzzle_anim = True
                st.session_state.puzzle_instant_mode = False
            else:
                st.session_state.trigger_puzzle_anim = False
                st.session_state.puzzle_instant_mode = True
            st.rerun()

    with p_col2:
        p_autoplay_now = bool(st.session_state.trigger_puzzle_anim)
        if p_autoplay_now:
            st.session_state.trigger_puzzle_anim = False
            p_instant_now = False
        else:
            p_instant_now = bool(st.session_state.puzzle_results) or st.session_state.puzzle_instant_mode

        puzzle_html = build_puzzle_animator_html(
            start_state=st.session_state.puzzle_state,
            goal_state=st.session_state.puzzle_goal,
            results=st.session_state.puzzle_results,
            autoplay=p_autoplay_now,
            instant=p_instant_now,
            canvas_px=340,
            run_id=st.session_state.puzzle_solve_seq,
        )
        st.iframe(puzzle_html, width="stretch", height=495)

    if st.session_state.puzzle_results:
        solved_names = [n for n, r in st.session_state.puzzle_results.items() if r.found]
        if solved_names:
            with st.expander("📜 View Step-by-Step Move Log", expanded=False):
                if len(solved_names) > 1:
                    log_algo = st.selectbox(
                        "Show Move Sequence For:",
                        solved_names,
                        index=0,
                        key="puzzle_log_algo_sel",
                    )
                else:
                    log_algo = solved_names[0]

                log_res = st.session_state.puzzle_results[log_algo]
                total_moves = len(log_res.path) - 1
                if total_moves == 0:
                    st.info("Board is already in the goal state (0 moves required).")
                else:
                    dir_styles = {
                        "UP": ("⬆️ UP", "#8be9fd"),
                        "DOWN": ("⬇️ DOWN", "#ffb86c"),
                        "LEFT": ("⬅️ LEFT", "#bd93f9"),
                        "RIGHT": ("➡️ RIGHT", "#f1fa8c"),
                    }
                    max_display = 200
                    show_truncated = total_moves > max_display
                    step_indices = list(range(1, min(total_moves, max_display) + 1))

                    rows_html = []
                    for idx in step_indices:
                        prev_b = log_res.path[idx - 1]
                        curr_b = log_res.path[idx]
                        prev_zero = prev_b.index(0)
                        curr_zero = curr_b.index(0)
                        moved_t = prev_b[curr_zero]
                        r0, c0 = divmod(curr_zero, 3)
                        r1, c1 = divmod(prev_zero, 3)
                        dr, dc = r1 - r0, c1 - c0
                        direction = (
                            "UP"
                            if dr == -1
                            else ("DOWN" if dr == 1 else ("LEFT" if dc == -1 else "RIGHT"))
                        )
                        dir_lbl, dir_clr = dir_styles[direction]
                        board_str = " ".join(str(x) if x != 0 else "·" for x in curr_b[0:3]) + " | " + " ".join(str(x) if x != 0 else "·" for x in curr_b[3:6]) + " | " + " ".join(str(x) if x != 0 else "·" for x in curr_b[6:9])
                        goal_tag = ' <span style="color:#50fa7b; font-weight:700; margin-left:6px;">🎉 Goal</span>' if idx == total_moves else ""
                        rows_html.append(
                            f"""
                            <div style="display:grid; grid-template-columns:56px 78px 92px 1fr 190px; align-items:center; gap:8px; padding:6px 10px; border-bottom:1px solid #343746; font-size:0.84rem;">
                                <span style="color:#bd93f9; font-weight:700; font-family:monospace;">#{idx:02d}</span>
                                <span><span style="background:#ff79c6; color:#282a36; font-weight:800; padding:2px 8px; border-radius:4px;">Tile {moved_t}</span></span>
                                <span style="color:{dir_clr}; font-weight:700;">{dir_lbl}</span>
                                <span style="color:#f8f8f2;">({r0},{c0}) → ({r1},{c1}){goal_tag}</span>
                                <span style="color:#8be9fd; font-family:monospace; font-size:0.78rem; background:#282a36; padding:2px 6px; border-radius:4px; text-align:center;">{board_str}</span>
                            </div>
                            """
                        )

                    trunc_note = (
                        f'<div style="padding:6px 10px; color:#ffb86c; font-size:0.8rem;">Showing first {max_display} of {total_moves} total moves.</div>'
                        if show_truncated
                        else ""
                    )
                    log_html = f"""
                    <div style="border:1px solid #44475a; border-radius:8px; background:#21222c; overflow:hidden; margin-bottom:6px;">
                        <div style="display:grid; grid-template-columns:56px 78px 92px 1fr 190px; gap:8px; padding:7px 10px; background:#282a36; border-bottom:1px solid #6272a4; font-size:0.78rem; font-weight:700; color:#8be9fd; text-transform:uppercase;">
                            <span>Step</span><span>Tile</span><span>Direction</span><span>Move Coordinates</span><span style="text-align:center;">Board Snapshot</span>
                        </div>
                        <div style="max-height:260px; overflow-y:auto;">
                            {''.join(rows_html)}
                            {trunc_note}
                        </div>
                    </div>
                    """
                    st.markdown(log_html, unsafe_allow_html=True)

        if len(st.session_state.puzzle_results) > 1:
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
                st.dataframe(pd.DataFrame(p_table_rows), width="stretch", hide_index=True)


# =========================================================
# VIEW 3: Algorithm Theory & Complexity
# =========================================================
elif nav_selection == "3. Algorithm Theory & Complexity":
    st.markdown(
        """
        <div class="header-bar">
            <span class="header-title">Search Algorithm Theory & Complexity</span>
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
    st.dataframe(pd.DataFrame(theory_data), width="stretch", hide_index=True)

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
