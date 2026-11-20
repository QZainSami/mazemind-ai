"""
MazeMind AI: Intelligent Puzzle Solver & Search Algorithm Visualizer.
Course: CSL-325 Artificial Intelligence Lab (Fall 2026).
Section: BS (CS) 5 - B | Course Instructor: Sir Shahid.

Group Members:
- Samiullah Qureshi (02-134242-111) — System architecture, OOP design, Greedy Best-First, A*, Cloud deployment
- Muhammad Musavir Waseem (02-134242-008) — BFS, DFS, 8-puzzle state-space logic, solvability validation, grid UI
- Farman Ali (02-134241-015) — Step visualization, comparison charts/tables, unit testing, documentation
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
)

# ---------------------------------------------------------
# Page Configuration & Global Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="MazeMind AI — Puzzle Solver & Search Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #546E7A;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .team-badge {
        background: linear-gradient(135deg, #E3F2FD, #EDE7F6);
        border: 1px solid #BBDEFB;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 14px;
        font-size: 0.88rem;
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

if "puzzle_state" not in st.session_state:
    st.session_state.puzzle_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)

if "puzzle_goal" not in st.session_state:
    st.session_state.puzzle_goal = EightPuzzle.STANDARD_GOAL

if "puzzle_results" not in st.session_state:
    st.session_state.puzzle_results = {}

# ---------------------------------------------------------
# Sidebar: Navigation & Project Profile
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 MazeMind AI")
    st.caption("Intelligent Puzzle Solver & Search Algorithm Visualizer")
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
    st.markdown("#### 👥 Group Profile (BS CS 5-B)")
    st.markdown(
        """
        <div class="team-badge">
            <b>Course:</b> CSL-325 AI Lab (Fall 2026)<br>
            <b>Instructor:</b> Sir Shahid<br><br>
            <b>1. Samiullah Qureshi</b> (02-134242-111)<br>
            <b>2. Muhammad Musavir Waseem</b> (02-134242-008)<br>
            <b>3. Farman Ali</b> (02-134241-015)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📌 Key Principles")
    st.caption(
        "• **Zero Pre-Trained Models**: Solvers built strictly from scratch in pure Python using OOP.\n"
        "• **Uninformed Search**: BFS & DFS with cycle prevention.\n"
        "• **Informed Search**: Greedy & A* with Manhattan & Euclidean heuristics.\n"
        "• **Inversion Parity Theorem** guaranteed for 8-Puzzle solvability."
    )


# =========================================================
# VIEW 1: 2D Grid Maze Solver & Visualizer
# =========================================================
if nav_selection == "1. 2D Grid Maze Solver":
    st.markdown('<div class="main-header">2D Grid Maze Solver & Visualizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Design custom mazes, compare Uninformed (BFS, DFS) vs. Informed (Greedy, A*) search, and watch step-by-step exploration animations.</div>',
        unsafe_allow_html=True,
    )

    col_ctrl, col_view = st.columns([1, 1.4])

    with col_ctrl:
        st.markdown("#### 🗺️ Maze Layout Controls")

        c_size, c_dens = st.columns(2)
        with c_size:
            dim = st.slider(
                "Grid Dimension (N×N):",
                min_value=8,
                max_value=35,
                value=st.session_state.grid_size,
                help="Bounded up to 35x35 per proposal constraint for responsive browser execution.",
            )
        with c_dens:
            density = st.slider("Target Obstacle Density:", 0.05, 0.45, 0.22, 0.01)

        preset = st.selectbox(
            "Pattern Preset:",
            [
                "Random Obstacles",
                "Open Room (Zero Walls)",
                "Center Wall Barrier",
                "Snake Labyrinth",
                "Current Grid",
            ],
        )

        if st.button("Generate / Reset Maze", use_container_width=True):
            st.session_state.grid_size = dim
            st.session_state.search_results = {}

            if preset == "Open Room (Zero Walls)":
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
            else:  # Random Obstacles or Current Grid reset
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
            <div class="metric-card">
                <b>Maze Statistics:</b><br>
                • Dimensions: <b>{stats['rows']} × {stats['cols']}</b> ({stats['total_cells']} total cells)<br>
                • Wall Density: <b>{stats['wall_density']*100:.1f}%</b> ({stats['wall_cells']} walls, {stats['open_cells']} open)<br>
                • Distance to Goal: Manhattan = <b>{stats['manhattan_dist']}</b>, Euclidean = <b>{stats['euclidean_dist']:.2f}</b><br>
                • Dead-End Terminals: <b>{stats['dead_end_count']}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### ⚡ Solver & Heuristic Setup")

        heuristic_choice = st.radio(
            "Heuristic Metric for Informed Solvers:",
            ["Manhattan Distance (|Δr| + |Δc|)", "Euclidean Distance (Straight-Line)"],
            horizontal=True,
        )
        metric_key = "manhattan" if "Manhattan" in heuristic_choice else "euclidean"

        solver_selection = st.selectbox(
            "Search Algorithm(s) to Run:",
            [
                "Run All 4 (Side-by-Side Comparison)",
                "A* Search",
                "Breadth-First Search (BFS)",
                "Depth-First Search (DFS)",
                "Greedy Best-First Search",
            ],
        )

        if st.button("Solve Maze", type="primary", use_container_width=True):
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

    with col_view:
        st.markdown("#### 🎨 Live Maze Visualization & Animation")

        if st.session_state.search_results:
            agent_names = list(st.session_state.search_results.keys())
            selected_agent_name = st.selectbox(
                "Display Exploration Playback For:",
                agent_names,
                index=0,
            )
            active_res = st.session_state.search_results[selected_agent_name]

            if active_res.found:
                step_count = len(active_res.exploration_history)

                # Animation Speed and Playback Controls
                anim_c1, anim_c2 = st.columns([1.2, 1])
                with anim_c1:
                    anim_speed = st.select_slider(
                        "⚡ Animation Speed:",
                        options=["0.25x (Slow-Mo)", "0.5x", "1.0x (Normal)", "2.0x (Fast)", "5.0x (Turbo)"],
                        value="1.0x (Normal)",
                    )
                with anim_c2:
                    st.write("")
                    st.write("")
                    play_anim = st.button("▶️ Play Animation", type="secondary", use_container_width=True)

                plot_slot = st.empty()
                progress_slot = st.empty()

                if play_anim:
                    speed_map = {
                        "0.25x (Slow-Mo)": (0.12, 1),
                        "0.5x": (0.05, 1),
                        "1.0x (Normal)": (0.02, 2),
                        "2.0x (Fast)": (0.01, 4),
                        "5.0x (Turbo)": (0.005, 8),
                    }
                    delay, stride = speed_map[anim_speed]

                    for s in range(1, step_count + 1, stride):
                        fig = plot_maze(
                            curr_maze,
                            result=active_res,
                            step=s,
                            title=f"{selected_agent_name} Expanding... (Step {s}/{step_count})",
                        )
                        plot_slot.pyplot(fig)
                        plt.close(fig)
                        progress_slot.progress(
                            s / step_count, text=f"Exploring state space: {s}/{step_count} nodes"
                        )
                        time.sleep(delay)

                    fig = plot_maze(
                        curr_maze,
                        result=active_res,
                        step=step_count,
                        title=f"✅ {selected_agent_name} Complete! Optimal Path: {active_res.path_cost} Steps",
                    )
                    plot_slot.pyplot(fig)
                    plt.close(fig)
                    progress_slot.empty()
                else:
                    step = st.slider(
                        "Manual Step-by-Step Scrubber:",
                        min_value=1,
                        max_value=max(1, step_count),
                        value=step_count,
                        help="Slide to manually inspect the wave of visited nodes.",
                    )
                    fig = plot_maze(
                        curr_maze,
                        result=active_res,
                        step=step,
                        title=f"{selected_agent_name} (Step {step}/{step_count} — Path Cost: {active_res.path_cost})",
                    )
                    plot_slot.pyplot(fig)
                    plt.close(fig)
            else:
                st.error("❌ No path found between Start (0, 0) and Goal!")
                fig = plot_maze(curr_maze, result=active_res, title=f"{selected_agent_name} (Unsolvable)")
                st.pyplot(fig)
                plt.close(fig)
        else:
            fig = plot_maze(curr_maze, title=f"Initial Grid ({curr_maze.rows}×{curr_maze.cols})")
            st.pyplot(fig)
            plt.close(fig)

    # -----------------------------------------------------
    # Performance Comparison Section (Tables & Bar Charts)
    # -----------------------------------------------------
    if st.session_state.search_results:
        st.markdown("---")
        st.markdown("### 📊 Search Performance Benchmarking")
        st.caption(
            "Side-by-side comparison illustrating path completeness, optimality, and node expansion trade-offs."
        )

        rows_data = []
        for name, res in st.session_state.search_results.items():
            rows_data.append(
                {
                    "Algorithm": name,
                    "Heuristic": res.heuristic_metric.title(),
                    "Status": "✅ Solved" if res.found else "❌ Failed",
                    "Path Length (Steps)": res.path_cost if res.found else "N/A",
                    "Nodes Expanded": res.nodes_expanded,
                    "Max Frontier Size": res.max_frontier_size,
                    "Execution Time (ms)": f"{res.execution_time_ms:.3f}",
                }
            )

        comp_df = pd.DataFrame(rows_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        if len(st.session_state.search_results) > 1:
            st.markdown("##### Comparative Performance Bar Charts")
            fig_bars = plot_comparison_bars(st.session_state.search_results)
            st.pyplot(fig_bars)
            plt.close(fig_bars)


# =========================================================
# VIEW 2: 8-Puzzle State-Space Solver & Visualizer
# =========================================================
elif nav_selection == "2. 8-Puzzle State-Space Solver":
    st.markdown('<div class="main-header">8-Puzzle State-Space Solver & Visualizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Solve 3×3 sliding tile puzzles using A* Search and Greedy Best-First Search with mathematical solvability validation.</div>',
        unsafe_allow_html=True,
    )

    p_col1, p_col2 = st.columns([1, 1.3])

    with p_col1:
        st.markdown("#### ⚙️ Initial Board Setup")

        preset_puzzle = st.selectbox(
            "Choose Initial Board Preset:",
            [
                "1-Move Trivial Solution",
                "3-Moves Easy",
                "6-Moves Medium",
                "10-Moves Challenging",
                "Generate Random Solvable Board",
            ],
        )

        goal_choice = st.selectbox(
            "Target Goal State:",
            [
                "Standard: (1, 2, 3, 4, 5, 6, 7, 8, 0)",
                "Spiral: (1, 2, 3, 8, 0, 4, 7, 6, 5)",
            ],
        )
        selected_goal = (
            EightPuzzle.STANDARD_GOAL
            if "Standard" in goal_choice
            else EightPuzzle.SPIRAL_GOAL
        )

        if st.button("Apply / Scramble Board", use_container_width=True):
            if preset_puzzle == "1-Move Trivial Solution":
                st.session_state.puzzle_state = (1, 2, 3, 4, 5, 6, 7, 0, 8)
            elif preset_puzzle == "3-Moves Easy":
                st.session_state.puzzle_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)
            elif preset_puzzle == "6-Moves Medium":
                st.session_state.puzzle_state = (1, 2, 3, 0, 4, 6, 7, 5, 8)
            elif preset_puzzle == "10-Moves Challenging":
                st.session_state.puzzle_state = (1, 3, 6, 4, 2, 5, 7, 0, 8)
            else:  # Random Solvable Board
                while True:
                    cand = list(range(9))
                    random.shuffle(cand)
                    cand_tuple = tuple(cand)
                    if EightPuzzle.is_solvable(cand_tuple, selected_goal):
                        st.session_state.puzzle_state = cand_tuple
                        break

            st.session_state.puzzle_goal = selected_goal
            st.session_state.puzzle_results = {}
            st.rerun()

        # Inversion parity calculation
        inv_count = EightPuzzle.count_inversions(st.session_state.puzzle_state)
        goal_inv = EightPuzzle.count_inversions(st.session_state.puzzle_goal)
        is_solv = (inv_count % 2) == (goal_inv % 2)

        st.markdown(
            f"""
            <div class="metric-card">
                <b>Solvability Verification (Inversion Parity Theorem):</b><br>
                • Start Inversions: <b>{inv_count}</b> ({'Even' if inv_count % 2 == 0 else 'Odd'})<br>
                • Goal Inversions: <b>{goal_inv}</b> ({'Even' if goal_inv % 2 == 0 else 'Odd'})<br>
                • Solvability Status: <b>{'🟢 Solvable' if is_solv else '🔴 Unsolvable'}</b><br>
                <span style="font-size:0.8rem; color:#607D8B;">
                Theorem: On odd grid width (W=3), horizontal/vertical blank swaps preserve inversion parity.
                A board is solvable if and only if Inv(S) ≡ Inv(G) (mod 2).
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### ⚡ Heuristic & Solver Execution")
        p_metric = st.radio(
            "Heuristic Evaluation Function:",
            ["Manhattan Distance", "Euclidean Distance", "Misplaced Tiles"],
            horizontal=True,
        )
        metric_key = (
            "manhattan"
            if "Manhattan" in p_metric
            else ("euclidean" if "Euclidean" in p_metric else "misplaced")
        )

        p_solver_choice = st.selectbox(
            "Search Strategy:",
            [
                "Run All Solvers (Comparison)",
                f"A* Search ({metric_key.title()})",
                f"Greedy Best-First ({metric_key.title()})",
                "Breadth-First Search (BFS)",
                "Depth-First Search (DFS)",
            ],
        )

        if st.button("Solve 8-Puzzle", type="primary", use_container_width=True):
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
                    f"A* Search ({metric_key.title()})": AStarAgent(name=f"A* ({metric_key.title()})", heuristic_metric=metric_key),
                    f"Greedy Best-First ({metric_key.title()})": GreedyAgent(name=f"Greedy ({metric_key.title()})", heuristic_metric=metric_key),
                    "Breadth-First Search (BFS)": BFSAgent(name="BFS"),
                    "Depth-First Search (DFS)": DFSAgent(name="DFS"),
                }
                agents = [agent_dict[p_solver_choice]]

            p_results = {}
            with st.spinner("Searching puzzle state space..."):
                for ag in agents:
                    p_results[ag.name] = ag.solve(env_puzzle)
            st.session_state.puzzle_results = p_results

    with p_col2:
        st.markdown("#### 🎯 Board State Visualization & Tile Animation")

        if st.session_state.puzzle_results:
            p_names = list(st.session_state.puzzle_results.keys())
            active_p_name = st.selectbox("Select Solution to Animate:", p_names, index=0)
            res_p = st.session_state.puzzle_results[active_p_name]

            if res_p and res_p.found:
                st.success(
                    f"Solution Found by {active_p_name}! Moves: **{res_p.path_cost}** | Nodes: **{res_p.nodes_expanded}** | Time: **{res_p.execution_time_ms:.2f} ms**"
                )

                p_anim_c1, p_anim_c2 = st.columns([1.2, 1])
                with p_anim_c1:
                    puzzle_speed = st.select_slider(
                        "⚡ Move Speed:",
                        options=["Slow (0.6s)", "Normal (0.3s)", "Fast (0.15s)"],
                        value="Normal (0.3s)",
                    )
                with p_anim_c2:
                    st.write("")
                    st.write("")
                    play_puzzle_anim = st.button("▶️ Animate Tile Moves", type="secondary", use_container_width=True)

                p_slot = st.empty()

                if play_puzzle_anim:
                    p_delay = {"Slow (0.6s)": 0.6, "Normal (0.3s)": 0.3, "Fast (0.15s)": 0.15}[puzzle_speed]
                    for idx, step_board in enumerate(res_p.path):
                        fig = plot_eight_puzzle(
                            step_board,
                            title=f"Step {idx}/{len(res_p.path) - 1} ({'Goal Reached!' if idx == len(res_p.path) - 1 else 'Sliding Tiles...'})",
                        )
                        p_slot.pyplot(fig)
                        plt.close(fig)
                        time.sleep(p_delay)
                else:
                    step_idx = st.slider(
                        "Manual Step Scrubber:",
                        0,
                        len(res_p.path) - 1,
                        0,
                        help="Slide to inspect tile transitions step-by-step.",
                    )
                    curr_board = res_p.path[step_idx]
                    fig = plot_eight_puzzle(
                        curr_board,
                        title=f"Step {step_idx} of {len(res_p.path) - 1} ({'Goal Reached!' if step_idx == len(res_p.path) - 1 else 'In Transit'})",
                    )
                    p_slot.pyplot(fig)
                    plt.close(fig)
            else:
                st.error("No path found to goal.")
                fig = plot_eight_puzzle(st.session_state.puzzle_state, title="Initial State (Unsolvable)")
                st.pyplot(fig)
                plt.close(fig)
        else:
            fig = plot_eight_puzzle(st.session_state.puzzle_state, title="Current Initial State")
            st.pyplot(fig)
            plt.close(fig)

    # Comparison Table for 8-Puzzle
    if st.session_state.puzzle_results:
        st.markdown("---")
        st.markdown("### 📊 8-Puzzle Search Performance Comparison")
        p_table_rows = []
        for name, res in st.session_state.puzzle_results.items():
            p_table_rows.append(
                {
                    "Algorithm": name,
                    "Status": "✅ Solved" if res.found else "❌ Failed",
                    "Path Length (Moves)": res.path_cost if res.found else "N/A",
                    "Nodes Expanded": res.nodes_expanded,
                    "Max Frontier Size": res.max_frontier_size,
                    "Execution Time (ms)": f"{res.execution_time_ms:.3f}",
                }
            )
        st.dataframe(pd.DataFrame(p_table_rows), use_container_width=True, hide_index=True)

        if len(st.session_state.puzzle_results) > 1:
            fig_p_bars = plot_comparison_bars(st.session_state.puzzle_results)
            st.pyplot(fig_p_bars)
            plt.close(fig_p_bars)


# =========================================================
# VIEW 3: Algorithm Theory & Complexity
# =========================================================
elif nav_selection == "3. Algorithm Theory & Complexity":
    st.markdown('<div class="main-header">Search Algorithm Theory & Complexity Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Comprehensive theoretical comparison of classical Uninformed and Informed heuristic search algorithms.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Comparative Complexity Matrix")
    theory_data = [
        {
            "Algorithm": "Breadth-First Search (BFS)",
            "Strategy Type": "Uninformed (Blind)",
            "Evaluation Function": "g(n) = depth(n)",
            "Data Structure": "FIFO Queue (collections.deque)",
            "Time Complexity": "O(b^d)",
            "Space Complexity": "O(b^d)",
            "Complete?": "Yes (finite b)",
            "Optimal?": "Yes (unit cost)",
        },
        {
            "Algorithm": "Depth-First Search (DFS)",
            "Strategy Type": "Uninformed (Blind)",
            "Evaluation Function": "LIFO Stack order",
            "Data Structure": "Explicit Stack (list)",
            "Time Complexity": "O(b^m)",
            "Space Complexity": "O(b · m)",
            "Complete?": "Yes (with cycle check)",
            "Optimal?": "No",
        },
        {
            "Algorithm": "Greedy Best-First Search",
            "Strategy Type": "Informed (Heuristic)",
            "Evaluation Function": "f(n) = h(n)",
            "Data Structure": "Priority Queue (heapq)",
            "Time Complexity": "O(b^m) worst-case",
            "Space Complexity": "O(b^m)",
            "Complete?": "Yes (with visited check)",
            "Optimal?": "No",
        },
        {
            "Algorithm": "A* Search",
            "Strategy Type": "Informed (Optimal)",
            "Evaluation Function": "f(n) = g(n) + h(n)",
            "Data Structure": "Priority Queue (heapq)",
            "Time Complexity": "O(b^d) to O(b^m)",
            "Space Complexity": "O(b^d)",
            "Complete?": "Yes",
            "Optimal?": "Yes (admissible h)",
        },
    ]
    st.dataframe(pd.DataFrame(theory_data), use_container_width=True, hide_index=True)
    st.caption("Notation: b = branching factor, d = depth of shallowest goal, m = maximum depth of search space.")

    st.markdown("---")
    st.markdown("### 2. Heuristic Functions & Mathematical Guarantees")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown(
            r"""
            #### A. Manhattan Distance Heuristic ($L_1$ Norm)
            For orthogonal 4-directional grids, moving diagonally requires at least 2 steps:
            $$h_{\text{Manhattan}}(n) = |r_n - r_{\text{goal}}| + |c_n - c_{\text{goal}}|$$
            * **Admissibility:** Never overestimates the true remaining cost ($h(n) \le h^*(n)$) because it represents the distance without obstacles.
            * **Consistency (Monotonicity):** For any step from $n$ to $n'$ with step cost $c(n, n') = 1$:
              $$h(n) \le c(n, n') + h(n')$$
            """
        )

    with col_h2:
        st.markdown(
            r"""
            #### B. Euclidean Distance Heuristic ($L_2$ Norm)
            Calculates the straight-line geometric distance:
            $$h_{\text{Euclidean}}(n) = \sqrt{(r_n - r_{\text{goal}})^2 + (c_n - c_{\text{goal}})^2}$$
            * **Admissibility:** Since a straight line is the shortest path in Euclidean space, $h(n) \le h^*(n)$ holds strictly.
            * **Domination Comparison:** On 4-directional grids, $h_{\text{Manhattan}}(n) \ge h_{\text{Euclidean}}(n)$ everywhere. Hence, Manhattan dominates Euclidean and expands fewer or equal nodes in A*.
            """
        )

    st.markdown("---")
    st.markdown("### 3. Inversion Parity Theorem (8-Puzzle Solvability)")
    st.markdown(
        r"""
        The 8-puzzle board consists of 9 cells containing numbers $1..8$ and a blank ($0$).
        An **inversion** occurs when a tile appears before another tile with a smaller number:
        $$\mathrm{Inv}(S) = \sum_{1 \le i < j \le 8} \mathbb{I}[S_i > S_j \land S_i \ne 0 \land S_j \ne 0]$$

        * **Horizontal Tile Slide:** Does not change the linear order of numbered tiles; inversions remain invariant ($\Delta \mathrm{Inv} = 0$).
        * **Vertical Tile Slide:** Moves a tile across exactly 2 intermediate positions. The tile swaps places with 2 other numbered tiles, changing the inversion count by $+2$, $0$, or $-2$.
        * **Invariant Conclusion:** The parity of the inversion count is an invariant ($\Delta \mathrm{Inv} \equiv 0 \pmod 2$). Therefore, an initial state $S$ can reach goal $G$ if and only if:
          $$\mathrm{Inv}(S) \equiv \mathrm{Inv}(G) \pmod 2$$
        """
    )
