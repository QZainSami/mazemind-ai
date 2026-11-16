"""
MazeMind AI: Intelligent Puzzle Solver, Evolutionary Level Generator & Search Performance Predictor.
Course: CSL-325 Artificial Intelligence Lab (Fall 2026).
Interactive multi-tab Streamlit application.
"""

import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Add src to sys.path for local and cloud imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

try:
    from mazemind.environments import GridMaze, EightPuzzle
    from mazemind.search_agents import (
        SearchResult,
        BFSAgent,
        DFSAgent,
        GreedyAgent,
        AStarAgent,
    )
    from mazemind.genetic_generator import LevelEvolver
    from mazemind.ml_predictor import MLPredictor, FEATURE_COLUMNS
    from mazemind.visualizer import (
        plot_maze,
        plot_eight_puzzle,
        plot_ga_convergence,
        plot_confusion_matrix_heatmap,
        plot_decision_tree_graph,
        plot_actual_vs_predicted,
        plot_correlation_heatmap,
    )
except ImportError:
    from src.mazemind.environments import GridMaze, EightPuzzle
    from src.mazemind.search_agents import (
        SearchResult,
        BFSAgent,
        DFSAgent,
        GreedyAgent,
        AStarAgent,
    )
    from src.mazemind.genetic_generator import LevelEvolver
    from src.mazemind.ml_predictor import MLPredictor, FEATURE_COLUMNS
    from src.mazemind.visualizer import (
        plot_maze,
        plot_eight_puzzle,
        plot_ga_convergence,
        plot_confusion_matrix_heatmap,
        plot_decision_tree_graph,
        plot_actual_vs_predicted,
        plot_correlation_heatmap,
    )

# ---------------------------------------------------------
# Page Configuration & Global Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="MazeMind AI — Intelligent Puzzle Solver & Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished lab UI
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
    .prediction-banner {
        background: linear-gradient(135deg, #E3F2FD, #EDE7F6);
        border: 1px solid #BBDEFB;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0 18px 0;
    }
    .algo-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .algo-astar { background-color: #E8F5E9; color: #2E7D32; }
    .algo-bfs { background-color: #E3F2FD; color: #1565C0; }
    .algo-greedy { background-color: #FFF3E0; color: #E65100; }
    .algo-dfs { background-color: #F3E5F5; color: #6A1B9A; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_ml_predictor() -> MLPredictor:
    """Cached loader for MLPredictor."""
    return MLPredictor(models_dir="models")


@st.cache_data
def load_benchmark_data() -> pd.DataFrame:
    """Loads and caches the benchmark CSV."""
    if os.path.exists("data/maze_benchmarks.csv"):
        return pd.read_csv("data/maze_benchmarks.csv")
    return pd.DataFrame()


predictor = get_ml_predictor()
df_benchmarks = load_benchmark_data()

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "grid_size" not in st.session_state:
    st.session_state.grid_size = 12

if "arena_grid" not in st.session_state:
    # Default 12x12 maze with moderate walls
    rng = np.random.default_rng(42)
    grid = (rng.random((12, 12)) < 0.22).astype(np.int32)
    grid[0, 0] = 0
    grid[11, 11] = 0
    st.session_state.arena_grid = grid

if "search_results" not in st.session_state:
    st.session_state.search_results = {}

if "active_agent_result" not in st.session_state:
    st.session_state.active_agent_result = None

if "puzzle_state" not in st.session_state:
    st.session_state.puzzle_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)

if "puzzle_goal" not in st.session_state:
    st.session_state.puzzle_goal = EightPuzzle.STANDARD_GOAL

if "puzzle_result" not in st.session_state:
    st.session_state.puzzle_result = None

if "evolved_maze" not in st.session_state:
    st.session_state.evolved_maze = None

if "evolved_history" not in st.session_state:
    st.session_state.evolved_history = None

# ---------------------------------------------------------
# Sidebar: Global Navigation & Academic Context
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 MazeMind AI")
    st.caption("CSL-325 Artificial Intelligence Lab")
    st.markdown("---")

    st.markdown("#### 🎯 Navigation")
    tabs_menu = [
        "1. Search & Puzzle Arena",
        "2. Evolutionary Level Studio",
        "3. Dataset & EDA Explorer",
        "4. ML Training & Evaluation",
    ]
    selected_tab = st.radio("Go to view:", tabs_menu, index=0)

    st.markdown("---")
    st.markdown("#### ⚙️ System Status")
    models_ready = predictor.is_ready()
    st.write(f"• **ML Models:** {'🟢 Loaded' if models_ready else '🔴 Missing'}")
    st.write(
        f"• **Benchmark Data:** {'🟢 1,500 Mazes' if not df_benchmarks.empty else '🔴 Missing'}"
    )
    st.write("• **Optimization:** uv & NumPy")

    st.markdown("---")
    st.markdown("#### 📖 Project Highlights")
    st.caption(
        "• **Zero Pre-Trained Models**: Solvers, Genetic Algorithm, and ML pipelines trained 100% from scratch.\n"
        "• **Inversion Parity Theorem** guaranteed for 8-Puzzle solvability.\n"
        "• **Decision Tree & Linear Regression** deployed as real-time inference pre-scan."
    )


# =========================================================
# TAB 1: Search & Puzzle Arena
# =========================================================
if selected_tab == "1. Search & Puzzle Arena":
    st.markdown('<div class="main-header">Search & Puzzle Arena</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Solve 2D Grid Mazes and 8-Puzzle sliding tiles using BFS, DFS, Greedy Best-First, and A* with real-time ML pre-scan.</div>',
        unsafe_allow_html=True,
    )

    arena_mode = st.radio(
        "Select Problem Domain:",
        ["2D Grid Maze", "8-Puzzle Sliding Tiles"],
        horizontal=True,
    )

    # -----------------------------------------------------
    # Domain A: 2D Grid Maze
    # -----------------------------------------------------
    if arena_mode == "2D Grid Maze":
        col_ctrl, col_view = st.columns([1, 1.4])

        with col_ctrl:
            st.markdown("#### 🗺️ Maze Configuration")

            preset = st.selectbox(
                "Choose Maze Pattern Preset:",
                [
                    "Current Maze",
                    "Random Wall Placement",
                    "Open Room (No Obstacles)",
                    "Center Wall Barrier",
                    "Snake Labyrinth",
                    "Import from Level Studio",
                ],
            )

            c_size, c_dens = st.columns(2)
            with c_size:
                grid_dim = st.slider("Grid Dimension (N×N):", 8, 25, st.session_state.grid_size)
            with c_dens:
                wall_pct = st.slider("Target Density:", 0.05, 0.40, 0.22, 0.01)

            if st.button("Generate / Refresh Maze", use_container_width=True):
                st.session_state.grid_size = grid_dim
                st.session_state.search_results = {}
                st.session_state.active_agent_result = None

                if preset == "Open Room (No Obstacles)":
                    new_grid = np.zeros((grid_dim, grid_dim), dtype=np.int32)
                elif preset == "Center Wall Barrier":
                    new_grid = np.zeros((grid_dim, grid_dim), dtype=np.int32)
                    mid = grid_dim // 2
                    new_grid[0 : grid_dim - 2, mid] = 1
                elif preset == "Snake Labyrinth":
                    new_grid = np.zeros((grid_dim, grid_dim), dtype=np.int32)
                    for r in range(2, grid_dim - 1, 3):
                        if (r // 3) % 2 == 0:
                            new_grid[r, 1:] = 1
                        else:
                            new_grid[r, :-1] = 1
                elif preset == "Import from Level Studio" and st.session_state.evolved_maze is not None:
                    new_grid = st.session_state.evolved_maze.grid.copy()
                    st.session_state.grid_size = new_grid.shape[0]
                else:  # Random Wall Placement
                    rng = np.random.default_rng()
                    new_grid = (rng.random((grid_dim, grid_dim)) < wall_pct).astype(np.int32)

                new_grid[0, 0] = 0
                new_grid[grid_dim - 1, grid_dim - 1] = 0
                st.session_state.arena_grid = new_grid
                st.rerun()

            curr_maze = GridMaze(
                grid=st.session_state.arena_grid,
                start=(0, 0),
                goal=(
                    st.session_state.arena_grid.shape[0] - 1,
                    st.session_state.arena_grid.shape[1] - 1,
                ),
            )
            maze_feat = curr_maze.extract_features()

            # ---------------------------------------------
            # Live ML Pre-Scan Banner
            # ---------------------------------------------
            if predictor.is_ready():
                prediction = predictor.predict_maze(curr_maze)
                rec_algo = prediction["recommended_algo"]
                pred_cost = prediction["predicted_path_cost"]
                probs = prediction["probabilities"]

                algo_badge_class = {
                    "A*": "algo-astar",
                    "BFS": "algo-bfs",
                    "Greedy": "algo-greedy",
                }.get(rec_algo, "algo-astar")

                st.markdown(
                    f"""
                    <div class="prediction-banner">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #1E88E5; text-transform: uppercase;">
                            🤖 Real-Time ML Pre-Scan Analysis
                        </div>
                        <div style="margin-top: 4px; font-size: 1.15rem; font-weight: 700;">
                            Recommended Algorithm: <span class="algo-pill {algo_badge_class}">{rec_algo}</span>
                        </div>
                        <div style="font-size: 0.95rem; color: #37474F; margin-top: 4px;">
                            • Predicted Optimal Path Cost: <b>{pred_cost} steps</b><br>
                            • Model Confidence: <b>{probs.get(rec_algo, 0.0) * 100:.1f}%</b>
                            (BFS: {probs.get('BFS', 0.0)*100:.0f}%, Greedy: {probs.get('Greedy', 0.0)*100:.0f}%, A*: {probs.get('A*', 0.0)*100:.0f}%)
                        </div>
                        <div style="font-size: 0.8rem; color: #78909C; margin-top: 6px;">
                            Input features: Density={maze_feat['wall_density']:.2f} | Manhattan={maze_feat['manhattan_dist']} | Dead Ends={maze_feat['dead_end_count']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("#### 🚀 Algorithm Execution")
            algo_choice = st.selectbox(
                "Select Algorithm to Solve:",
                ["Run All (Side-by-Side Comparison)", "A*", "BFS", "DFS", "Greedy Best-First"],
            )

            if st.button("Solve Maze", type="primary", use_container_width=True):
                agents_to_run = (
                    [AStarAgent(), BFSAgent(), DFSAgent(), GreedyAgent()]
                    if algo_choice.startswith("Run All")
                    else [
                        {
                            "A*": AStarAgent(),
                            "BFS": BFSAgent(),
                            "DFS": DFSAgent(),
                            "Greedy Best-First": GreedyAgent(),
                        }[algo_choice]
                    ]
                )

                results = {}
                for ag in agents_to_run:
                    res = ag.solve(curr_maze)
                    results[ag.name] = res

                st.session_state.search_results = results
                if algo_choice.startswith("Run All"):
                    st.session_state.active_agent_result = results.get("A*")
                else:
                    st.session_state.active_agent_result = results.get(
                        list(results.keys())[0]
                    )

        with col_view:
            st.markdown("#### 🎨 Live Maze Visualization & Animation")

            # Solver Selection for playback
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

                    # --- Animation Controls ---
                    anim_c1, anim_c2 = st.columns([1.2, 1])
                    with anim_c1:
                        anim_speed = st.select_slider(
                            "⚡ Playback Speed:",
                            options=["0.25x (Slow-Mo)", "0.5x", "1.0x (Normal)", "2.0x (Fast)", "5.0x (Turbo)"],
                            value="1.0x (Normal)",
                            help="Control how fast the wave of explored nodes spreads across the maze.",
                        )
                    with anim_c2:
                        st.write("")
                        st.write("")
                        play_anim = st.button("▶️ Play Animation", type="secondary", use_container_width=True)

                    plot_container = st.empty()
                    progress_container = st.empty()

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
                            plot_container.pyplot(fig)
                            plt.close(fig)
                            progress_container.progress(
                                s / step_count, text=f"Exploring state space: {s}/{step_count} nodes"
                            )
                            time.sleep(delay)

                        # Final Frame with complete path
                        fig = plot_maze(
                            curr_maze,
                            result=active_res,
                            step=step_count,
                            title=f"✅ {selected_agent_name} Complete! Path Cost: {active_res.path_cost} Steps",
                        )
                        plot_container.pyplot(fig)
                        plt.close(fig)
                        progress_container.empty()
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
                            title=f"{selected_agent_name} Search (Step {step}/{step_count} — Cost: {active_res.path_cost})",
                        )
                        plot_container.pyplot(fig)
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

        # -------------------------------------------------
        # Performance Comparison Table
        # -------------------------------------------------
        if st.session_state.search_results:
            st.markdown("---")
            st.markdown("### 📊 Benchmark Comparison Matrix")

            rows_data = []
            for name, res in st.session_state.search_results.items():
                rows_data.append(
                    {
                        "Algorithm": name,
                        "Status": "✅ Solved" if res.found else "❌ Failed",
                        "Path Cost (Steps)": res.path_cost if res.found else "N/A",
                        "Nodes Expanded": res.nodes_expanded,
                        "Max Frontier Size": res.max_frontier_size,
                        "Execution Time (ms)": f"{res.execution_time_ms:.3f}",
                    }
                )

            comp_df = pd.DataFrame(rows_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # -----------------------------------------------------
    # Domain B: 8-Puzzle Sliding Tiles
    # -----------------------------------------------------
    else:
        st.markdown("### 🧩 8-Puzzle (3×3 Sliding Tiles)")
        p_col1, p_col2 = st.columns([1, 1.3])

        with p_col1:
            st.markdown("#### ⚙️ Puzzle State Setup")
            preset_puzzle = st.selectbox(
                "Choose Solvable Configuration Preset:",
                [
                    "1-Move Trivial Solution",
                    "3-Moves Easy",
                    "6-Moves Medium",
                    "10-Moves Challenging",
                    "Generate Random Solvable Board",
                ],
            )

            target_goal_choice = st.selectbox(
                "Target Goal Board:",
                ["Standard: (1, 2, 3, 4, 5, 6, 7, 8, 0)", "Spiral: (1, 2, 3, 8, 0, 4, 7, 6, 5)"],
            )
            selected_goal = (
                EightPuzzle.STANDARD_GOAL
                if "Standard" in target_goal_choice
                else EightPuzzle.SPIRAL_GOAL
            )

            if st.button("Apply Board Preset", use_container_width=True):
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
                st.session_state.puzzle_result = None
                st.rerun()

            # Inversion parity solvability check
            inv_count = EightPuzzle.count_inversions(st.session_state.puzzle_state)
            is_solv = EightPuzzle.is_solvable(
                st.session_state.puzzle_state, st.session_state.puzzle_goal
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <b>Solvability Status:</b> {'🟢 Solvable' if is_solv else '🔴 Unsolvable'}<br>
                    • Inversion Count: <b>{inv_count}</b> ({'Even' if inv_count % 2 == 0 else 'Odd'})<br>
                    • Theorem: An 8-puzzle on odd width (3) is solvable if and only if initial parity matches goal parity.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### 🔍 Solve Configuration")
            p_algo = st.selectbox(
                "Search Algorithm:",
                ["A* (Manhattan)", "A* (Misplaced Tiles)", "BFS", "Greedy Best-First"],
            )

            if st.button("Solve 8-Puzzle", type="primary", use_container_width=True):
                env_puzzle = EightPuzzle(
                    start_state=st.session_state.puzzle_state,
                    goal_state=st.session_state.puzzle_goal,
                )

                if "Misplaced" in p_algo:
                    agent = AStarAgent(name="A* (Misplaced)", heuristic_metric="misplaced")
                elif "Greedy" in p_algo:
                    agent = GreedyAgent(name="Greedy Best-First", heuristic_metric="manhattan")
                elif "BFS" in p_algo:
                    agent = BFSAgent(name="BFS")
                else:
                    agent = AStarAgent(name="A* (Manhattan)", heuristic_metric="manhattan")

                with st.spinner("Searching state space..."):
                    p_res = agent.solve(env_puzzle)
                    st.session_state.puzzle_result = p_res

        with p_col2:
            st.markdown("#### 🎯 Board State Visualization & Animation")
            res_p = st.session_state.puzzle_result

            if res_p and res_p.found:
                st.success(
                    f"Solution Found! Path Cost: **{res_p.path_cost} moves** | Nodes Expanded: **{res_p.nodes_expanded}** | Time: **{res_p.execution_time_ms} ms**"
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
                fig = plot_eight_puzzle(
                    st.session_state.puzzle_state,
                    title="Current Initial State",
                )
                st.pyplot(fig)
                plt.close(fig)


# =========================================================
# TAB 2: Evolutionary Level Studio
# =========================================================
elif selected_tab == "2. Evolutionary Level Studio":
    st.markdown('<div class="main-header">Evolutionary Level Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Procedurally evolve challenging, solvable mazes tailored to a target path length using Genetic Algorithms.</div>',
        unsafe_allow_html=True,
    )

    c_evo_ctrl, c_evo_view = st.columns([1, 1.4])

    with c_evo_ctrl:
        st.markdown("#### 🧬 Genetic Algorithm Parameters")

        e_size = st.slider("Grid Size (Rows & Cols):", 10, 20, 15)
        e_target = st.slider("Target Path Length (Steps):", 15, 50, 25)
        e_pop = st.slider("Population Size:", 15, 60, 30)
        e_gens = st.slider("Generations:", 10, 60, 35)
        e_mut = st.slider("Mutation Rate:", 0.01, 0.15, 0.05, 0.01)

        st.caption(
            "$$\\text{Fitness} = 100 - 2\\cdot|L_{\\text{actual}} - L_{\\text{target}}| + 0.5\\cdot\\text{dead\\_ends}$$"
        )

        if st.button("Start Genetic Evolution", type="primary", use_container_width=True):
            evolver = LevelEvolver(
                rows=e_size,
                cols=e_size,
                target_path_length=e_target,
                population_size=e_pop,
                generations=e_gens,
                mutation_rate=e_mut,
                seed=random.randint(1, 99999),
            )

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_ui(gen, total_gens, best_fit, avg_fit):
                frac = gen / total_gens
                progress_bar.progress(frac)
                status_text.text(
                    f"Generation {gen}/{total_gens} — Best Fitness: {best_fit:.1f} | Avg Fitness: {avg_fit:.1f}"
                )

            with st.spinner("Evolving maze chromosomes..."):
                fittest_maze, history_df = evolver.evolve(progress_callback=update_ui)
                st.session_state.evolved_maze = fittest_maze
                st.session_state.evolved_history = history_df
                st.success("Evolution Complete!")

        if st.session_state.evolved_maze is not None:
            st.markdown("---")
            if st.button("🚀 Send Evolved Maze to Arena", type="secondary", use_container_width=True):
                st.session_state.arena_grid = st.session_state.evolved_maze.grid.copy()
                st.session_state.grid_size = st.session_state.evolved_maze.rows
                st.session_state.search_results = {}
                st.session_state.active_agent_result = None
                st.info("Evolved maze sent to Arena! Switch to Tab 1 to solve it.")

    with c_evo_view:
        if st.session_state.evolved_maze is not None:
            st.markdown("#### 🏆 Best Evolved Level")

            # Solve best maze to display path
            astar_checker = AStarAgent()
            check_res = astar_checker.solve(st.session_state.evolved_maze)

            fig_maze = plot_maze(
                st.session_state.evolved_maze,
                result=check_res,
                title=f"Evolved Maze (Path Length: {check_res.path_cost} | Target: {e_target})",
            )
            st.pyplot(fig_maze)
            plt.close(fig_maze)

            st.markdown("#### 📈 Fitness Convergence Curve")
            fig_curve = plot_ga_convergence(st.session_state.evolved_history, e_target)
            st.pyplot(fig_curve)
            plt.close(fig_curve)
        else:
            st.info("Configure GA parameters and click **Start Genetic Evolution** to generate a maze.")


# =========================================================
# TAB 3: Dataset & EDA Explorer
# =========================================================
elif selected_tab == "3. Dataset & EDA Explorer":
    st.markdown('<div class="main-header">Dataset & EDA Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Statistical profiling and exploratory data analysis of 1,500+ simulated maze benchmarks.</div>',
        unsafe_allow_html=True,
    )

    if df_benchmarks.empty:
        st.warning("Benchmark dataset not found at `data/maze_benchmarks.csv`.")
    else:
        # KPI metric cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Simulated Mazes", f"{len(df_benchmarks):,}")
        k2.metric("Recorded Features", len(FEATURE_COLUMNS))
        k3.metric("Missing / Null Values", int(df_benchmarks.isnull().sum().sum()))
        k4.metric("Solvability Guarantee", "100.0%")

        st.markdown("---")
        st.markdown("#### 🔍 Interactive Pandas `.query()` Filter")
        query_str = st.text_input(
            "Filter dataset with Pandas syntax (e.g., `wall_density > 0.25 and astar_path_cost > 30`):",
            value="",
        )

        filtered_df = df_benchmarks
        if query_str.strip():
            try:
                filtered_df = df_benchmarks.query(query_str)
                st.success(f"Matched **{len(filtered_df)}** / {len(df_benchmarks)} records.")
            except Exception as e:
                st.error(f"Invalid query syntax: {e}")
                filtered_df = df_benchmarks

        st.dataframe(
            filtered_df[
                [
                    "maze_id",
                    "rows",
                    "cols",
                    "wall_density",
                    "manhattan_dist",
                    "dead_end_count",
                    "astar_path_cost",
                    "astar_nodes_expanded",
                    "recommended_algo",
                ]
            ].head(100),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(f"Showing top {min(100, len(filtered_df))} rows.")

        # Download button
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name="mazemind_filtered_benchmarks.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("#### 📊 Exploratory Visualizations")

        eda1, eda2 = st.columns(2)

        with eda1:
            st.markdown("##### Recommended Algorithm Distribution")
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            class_counts = df_benchmarks["recommended_algo"].value_counts()
            colors = ["#FFB74D", "#81C784", "#64B5F6"]
            ax_bar.bar(class_counts.index, class_counts.values, color=colors[: len(class_counts)])
            ax_bar.set_ylabel("Count")
            for idx, val in enumerate(class_counts.values):
                ax_bar.text(idx, val + 15, str(val), ha="center", fontweight="bold")
            ax_bar.set_title("Ground-Truth Recommended Algorithm Counts")
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        with eda2:
            st.markdown("##### A* Path Cost Distribution")
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
            sns.histplot(
                df_benchmarks["astar_path_cost"],
                kde=True,
                color="#1976D2",
                bins=25,
                ax=ax_hist,
            )
            ax_hist.set_title("A* Path Cost Histogram with KDE")
            ax_hist.set_xlabel("Path Cost (Steps)")
            plt.tight_layout()
            st.pyplot(fig_hist)
            plt.close(fig_hist)

        st.markdown("##### Correlation Matrix Heatmap")
        corr_cols = FEATURE_COLUMNS + ["astar_path_cost", "astar_nodes_expanded"]
        fig_corr = plot_correlation_heatmap(df_benchmarks, corr_cols)
        st.pyplot(fig_corr)
        plt.close(fig_corr)

        st.markdown("##### Statistical Summary (`describe`)")
        st.dataframe(df_benchmarks[corr_cols].describe().T, use_container_width=True)


# =========================================================
# TAB 4: ML Training & Evaluation
# =========================================================
elif selected_tab == "4. ML Training & Evaluation":
    st.markdown('<div class="main-header">ML Training & Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Inspect model performance metrics, visualize Decision Tree logic, analyze Linear Regression weights, and test live predictions.</div>',
        unsafe_allow_html=True,
    )

    metrics = predictor.metrics
    dt_metrics = metrics.get("decision_tree", {})
    lr_metrics = metrics.get("linear_regression", {})

    # Top KPI Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decision Tree Accuracy", f"{dt_metrics.get('accuracy', 0.0) * 100:.2f}%")
    m2.metric("Linear Regression R²", f"{lr_metrics.get('r2_score', 0.0):.4f}")
    m3.metric("Regression RMSE", f"{lr_metrics.get('rmse', 0.0):.4f}")
    m4.metric("Regression MSE", f"{lr_metrics.get('mse', 0.0):.4f}")

    st.markdown("---")

    ml_tab1, ml_tab2, ml_tab3 = st.tabs(
        [
            "🌲 Model 1: Decision Tree Classifier",
            "📈 Model 2: Linear Regression",
            "🎛️ Interactive 'What-If' Predictor",
        ]
    )

    # Sub-Tab 1: Decision Tree
    with ml_tab1:
        st.markdown("#### Decision Tree Evaluation")
        col_cm, col_report = st.columns([1, 1.2])

        with col_cm:
            st.markdown("##### Confusion Matrix Heatmap")
            cm = dt_metrics.get("confusion_matrix", [])
            classes = dt_metrics.get("classes", [])
            acc = dt_metrics.get("accuracy", 0.0)
            if cm and classes:
                fig_cm = plot_confusion_matrix_heatmap(cm, classes, acc)
                st.pyplot(fig_cm)
                plt.close(fig_cm)

        with col_report:
            st.markdown("##### Classification Metrics")
            clf_rep = dt_metrics.get("classification_report", {})
            if clf_rep:
                rep_rows = []
                for k, v in clf_rep.items():
                    if isinstance(v, dict):
                        rep_rows.append(
                            {
                                "Class / Average": k,
                                "Precision": f"{v.get('precision', 0):.3f}",
                                "Recall": f"{v.get('recall', 0):.3f}",
                                "F1-Score": f"{v.get('f1-score', 0):.3f}",
                                "Support": int(v.get("support", 0)),
                            }
                        )
                st.dataframe(pd.DataFrame(rep_rows), use_container_width=True, hide_index=True)

            st.markdown("##### Feature Importances")
            f_imp = dt_metrics.get("feature_importances", {})
            if f_imp:
                imp_df = (
                    pd.DataFrame(list(f_imp.items()), columns=["Feature", "Importance"])
                    .sort_values(by="Importance", ascending=False)
                )
                st.dataframe(imp_df, use_container_width=True, hide_index=True)

        st.markdown("##### Decision Tree Graph (`plot_tree`)")
        if predictor.clf is not None:
            fig_tree = plot_decision_tree_graph(
                predictor.clf,
                feature_names=FEATURE_COLUMNS,
                class_names=predictor.clf.classes_,
            )
            st.pyplot(fig_tree)
            plt.close(fig_tree)

    # Sub-Tab 2: Linear Regression
    with ml_tab2:
        st.markdown("#### Linear Regression Performance")
        col_scatter, col_coef = st.columns([1.2, 1])

        with col_scatter:
            st.markdown("##### Actual vs. Predicted Path Cost")
            if not df_benchmarks.empty and predictor.reg is not None:
                X_all = df_benchmarks[FEATURE_COLUMNS]
                y_true = df_benchmarks["astar_path_cost"].to_numpy()
                y_pred = predictor.reg.predict(X_all)
                r2 = lr_metrics.get("r2_score", 0.0)
                rmse = lr_metrics.get("rmse", 0.0)

                # Sample 300 points for crisp plot
                sample_idx = np.random.choice(len(y_true), size=min(300, len(y_true)), replace=False)
                fig_act_pred = plot_actual_vs_predicted(
                    y_true[sample_idx], y_pred[sample_idx], r2, rmse
                )
                st.pyplot(fig_act_pred)
                plt.close(fig_act_pred)

        with col_coef:
            st.markdown("##### Learned Feature Weights (Coefficients)")
            coefs = lr_metrics.get("coefficients", {})
            if coefs:
                coef_df = pd.DataFrame(
                    list(coefs.items()), columns=["Feature", "Weight (β)"]
                ).sort_values(by="Weight (β)", ascending=False)
                st.dataframe(coef_df, use_container_width=True, hide_index=True)
                st.caption(f"Intercept ($β_0$): **{lr_metrics.get('intercept', 0.0)}**")

    # Sub-Tab 3: What-If Feature Explorer
    with ml_tab3:
        st.markdown("#### 🎛️ Live 'What-If' Scenario Predictor")
        st.caption("Adjust hypothetical maze geometry features to see real-time model inferences.")

        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            w_rows = st.slider("Rows:", 10, 30, 16)
            w_cols = st.slider("Cols:", 10, 30, 16)
            w_density = st.slider("Wall Density:", 0.05, 0.40, 0.20, 0.01)

        with w_col2:
            w_manh = st.slider("Manhattan Dist (Start to Goal):", 10, 60, (w_rows - 1) + (w_cols - 1))
            w_eucl = float(np.sqrt((w_rows - 1) ** 2 + (w_cols - 1) ** 2))
            w_dead_ends = st.slider("Dead End Count:", 0, 50, 8)
            w_avg_open = st.slider("Avg Open Neighbors:", 1.5, 4.0, 2.7, 0.1)

        with w_col3:
            w_total = w_rows * w_cols
            hypo_features = {
                "rows": w_rows,
                "cols": w_cols,
                "total_cells": w_total,
                "wall_density": w_density,
                "manhattan_dist": w_manh,
                "euclidean_dist": round(w_eucl, 4),
                "dead_end_count": w_dead_ends,
                "avg_open_neighbors": round(w_avg_open, 4),
            }

            if predictor.is_ready():
                hypo_pred = predictor.predict_features(hypo_features)
                st.markdown(
                    f"""
                    <div class="prediction-banner">
                        <div style="font-weight: 700; color: #1E88E5;">PREDICTED OUTCOME</div>
                        <h3 style="margin: 4px 0;">{hypo_pred['recommended_algo']}</h3>
                        <div>• Expected A* Cost: <b>{hypo_pred['predicted_path_cost']} steps</b></div>
                        <div>• Class Probabilities:</div>
                        <div style="font-size: 0.85rem; color: #555;">
                            {json.dumps(hypo_pred['probabilities'], indent=2)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("#### 🔄 Dynamic Re-Training")
        st.caption("Re-train both models dynamically on the 1,500 maze dataset and refresh serialized weights.")

        if st.button("Re-Train Models Now", type="primary"):
            from scripts.train_models import train_and_evaluate

            with st.spinner("Re-training Decision Tree & Linear Regression..."):
                updated_metrics = train_and_evaluate()
                predictor.load_models()
                st.success("Models re-trained and metrics re-saved successfully!")
                st.rerun()
