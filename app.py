"""
MazeMind AI: Intelligent Puzzle Solver & Search Algorithm Visualizer.
Theme: Official Dracula Theme (https://draculatheme.com/spec)
Features:
- Classical Search: BFS, DFS, UCS (Dijkstra), Greedy, A* & Weighted A*, Bidirectional BFS
- Environments: Guaranteed-solvable 2D GridMazes (Random, Labyrinths) & 8-Puzzle / 15-Puzzle
- Visuals: Synchronized 2x2 Race Mode, Open Frontier Highlights, Hover (g, h, f) Cell Inspector,
  Click-to-Slide Manual Tile Play, and Large High-DPI Canvases
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from mazemind.environments import GridMaze, EightPuzzle
from mazemind.search_agents import (
    BFSAgent,
    DFSAgent,
    UCSAgent,
    GreedyAgent,
    AStarAgent,
    BidirectionalBFSAgent,
)
from mazemind.visualizer import plot_comparison_bars
from mazemind.web_animator import (
    _MAZE_VER,
    _PUZZLE_VER,
    build_maze_payload,
    build_puzzle_payload,
    get_maze_component,
    get_puzzle_component,
)

# ---------------------------------------------------------
# Page Configuration & Dracula Global Styling
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
        padding-top: 0.8rem !important;
        padding-bottom: 0.4rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.6rem !important;
    }
    .header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 4px;
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
        padding: 5px 10px;
        color: #f8f8f2;
        font-size: 0.82rem;
        margin-bottom: 4px;
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
        gap: 0.32rem !important;
    }
    .stSelectbox, .stRadio, .stSlider {
        margin-bottom: 1px !important;
    }
    button[data-baseweb="tab"] {
        background-color: #21222c !important;
        border: 1px solid #44475a !important;
        border-radius: 8px 8px 0 0 !important;
        color: #f8f8f2 !important;
        font-weight: 600 !important;
        padding: 6px 14px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #282a36 !important;
        border-color: #bd93f9 !important;
        border-bottom-color: #bd93f9 !important;
        color: #bd93f9 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Register cached Streamlit v2 components
maze_component = get_maze_component()
puzzle_component = get_puzzle_component()

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
    init_m = GridMaze.generate_solvable(15, 15, density=0.22, preset="Random Obstacles", seed=42)
    st.session_state.maze_grid = init_m.grid

if "search_results" not in st.session_state:
    st.session_state.search_results = {}

if "trigger_maze_anim" not in st.session_state:
    st.session_state.trigger_maze_anim = False

if "maze_instant_mode" not in st.session_state:
    st.session_state.maze_instant_mode = False

if "maze_race_mode" not in st.session_state:
    st.session_state.maze_race_mode = True

if "maze_solve_seq" not in st.session_state:
    st.session_state.maze_solve_seq = 0

if "puzzle_size_label" not in st.session_state:
    st.session_state.puzzle_size_label = "3×3 (8-Puzzle)"

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

if "puzzle_race_mode" not in st.session_state:
    st.session_state.puzzle_race_mode = True

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
            "2. 8-Puzzle / 15-Puzzle Solver",
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

    col_ctrl, col_view = st.columns([0.85, 1.45])

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
            "Recursive Backtracker (Labyrinth)",
            "Prim's Perfect Maze",
            "Open Room (No Walls)",
            "Center Wall Barrier",
            "Snake Labyrinth",
        ]
        c_preset, c_gen_btn = st.columns([1.4, 0.9])
        with c_preset:
            preset = st.selectbox(
                "Preset Pattern:",
                preset_options,
                index=preset_options.index(st.session_state.grid_preset)
                if st.session_state.grid_preset in preset_options
                else 0,
                label_visibility="collapsed",
            )
        with c_gen_btn:
            new_maze_clicked = st.button("🎲 New Maze", width="stretch")

        params_changed = (
            dim != st.session_state.grid_size
            or abs(density - st.session_state.grid_density) > 1e-9
            or preset != st.session_state.grid_preset
        )

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
                📏 <b>{stats['rows']}×{stats['cols']}</b> ({stats['total_cells']} cells) |
                🧱 <b>{stats['wall_density']*100:.1f}% walls</b> |
                🎯 Dist: <b>{stats['manhattan_dist']:.0f}</b> |
                ✅ <b>Guaranteed Solvable</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### ⚡ Solver & Race Setup")

        c_alg, c_h = st.columns([1.35, 1])
        with c_alg:
            solver_selection = st.selectbox(
                "Search Strategy:",
                [
                    "🏁 Race Classical (A*, BFS, DFS, Greedy)",
                    "A* Search",
                    "Bidirectional BFS",
                    "UCS (Dijkstra)",
                    "Breadth-First Search (BFS)",
                    "Depth-First Search (DFS)",
                    "Greedy Best-First Search",
                ],
                index=0,
            )
        with c_h:
            uses_heuristic = any(
                k in solver_selection for k in ("A*", "Greedy", "Race")
            )
            heuristic_choice = st.selectbox(
                "Heuristic Metric:",
                ["Manhattan (L1)", "Euclidean (L2)"],
                disabled=not uses_heuristic,
            )
            metric_key = "manhattan" if "Manhattan" in heuristic_choice else "euclidean"

        uses_weight = any(k in solver_selection for k in ("A*", "Race"))
        astar_weight = st.slider(
            "Weighted A* Factor w (f = g + w·h):",
            min_value=1.0,
            max_value=3.0,
            value=1.0,
            step=0.25,
            disabled=not uses_weight,
            help="w = 1.0 guarantees optimal cost; w > 1.0 speeds up search.",
        )

        solve_mode = st.radio(
            "Execution Mode:",
            ["🎬 Animated (Step-by-Step)", "⚡ Instant Solution (No Animation)"],
            horizontal=True,
        )

        if st.button("🚀 Solve Maze", type="primary", width="stretch"):
            w_tag_str = f" (w={astar_weight:.2g})" if astar_weight > 1.0 else ""
            if solver_selection.startswith("🏁 Race"):
                agents = [
                    AStarAgent(
                        name=f"A*{w_tag_str}",
                        heuristic_metric=metric_key,
                        weight=astar_weight,
                    ),
                    BFSAgent(name="BFS"),
                    DFSAgent(name="DFS"),
                    GreedyAgent(name="Greedy", heuristic_metric=metric_key),
                ]
                is_race = True
            else:
                agent_map = {
                    "A* Search": AStarAgent(
                        name=f"A* ({metric_key.title()}){w_tag_str}",
                        heuristic_metric=metric_key,
                        weight=astar_weight,
                    ),
                    "Bidirectional BFS": BidirectionalBFSAgent(name="Bidirectional BFS"),
                    "UCS (Dijkstra)": UCSAgent(name="UCS (Dijkstra)"),
                    "Breadth-First Search (BFS)": BFSAgent(name="BFS"),
                    "Depth-First Search (DFS)": DFSAgent(name="DFS"),
                    "Greedy Best-First Search": GreedyAgent(
                        name=f"Greedy ({metric_key.title()})",
                        heuristic_metric=metric_key,
                    ),
                }
                agents = [agent_map[solver_selection]]
                is_race = False

            results = {}
            for ag in agents:
                results[ag.name] = ag.solve(curr_maze)
            st.session_state.search_results = results
            st.session_state.maze_race_mode = is_race
            st.session_state.maze_solve_seq += 1

            if "Animated" in solve_mode:
                st.session_state.trigger_maze_anim = True
                st.session_state.maze_instant_mode = False
            else:
                st.session_state.trigger_maze_anim = False
                st.session_state.maze_instant_mode = True
            st.rerun()

    with col_view:
        autoplay_now = bool(st.session_state.trigger_maze_anim)
        if autoplay_now:
            st.session_state.trigger_maze_anim = False
            instant_now = False
        else:
            instant_now = bool(st.session_state.search_results) or st.session_state.maze_instant_mode

        maze_payload = build_maze_payload(
            maze=curr_maze,
            results=st.session_state.search_results,
            autoplay=autoplay_now,
            instant=instant_now,
            race_mode=st.session_state.maze_race_mode,
            canvas_px=460,
            run_id=st.session_state.maze_solve_seq,
        )
        maze_component(data=maze_payload, key=f"maze_interactive_canvas_{_MAZE_VER}")

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
# VIEW 2: 8-Puzzle / 15-Puzzle Sliding Tile Solver
# =========================================================
elif nav_selection == "2. 8-Puzzle / 15-Puzzle Solver":
    st.markdown(
        """
        <div class="header-bar">
            <span class="header-title">8-Puzzle & 15-Puzzle Solver & Visualizer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p_col1, p_col2 = st.columns([0.85, 1.45])

    with p_col1:
        st.markdown("##### ⚙️ Puzzle Dimension & Initial Board")

        size_options = ["3×3 (8-Puzzle)", "4×4 (15-Puzzle)"]
        puzzle_size_label = st.radio(
            "Board Size:",
            size_options,
            index=size_options.index(st.session_state.puzzle_size_label)
            if st.session_state.puzzle_size_label in size_options
            else 0,
            horizontal=True,
        )
        is_15 = "4×4" in puzzle_size_label
        scramble_depth = 16 if is_15 else 18

        puzzle_presets = [
            "Random Solvable Board",
            "1-Move Trivial Solution",
            "3-Moves Easy",
            "6-Moves Medium",
            "10-Moves Challenging",
        ]
        goal_options = (
            ["Standard 4×4: (1..15, 0)"]
            if is_15
            else ["Standard: (1..8, 0)", "Spiral: (1,2,3,8,0,4,7,6,5)"]
        )

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
            goal_idx = (
                goal_options.index(st.session_state.puzzle_goal_label)
                if st.session_state.puzzle_goal_label in goal_options
                else 0
            )
            goal_choice = st.selectbox("Target Goal:", goal_options, index=goal_idx)

        selected_goal = (
            EightPuzzle.STANDARD_GOAL_15
            if is_15
            else (
                EightPuzzle.STANDARD_GOAL
                if "Standard" in goal_choice
                else EightPuzzle.SPIRAL_GOAL
            )
        )

        p_params_changed = (
            puzzle_size_label != st.session_state.puzzle_size_label
            or preset_puzzle != st.session_state.puzzle_preset
            or goal_choice != st.session_state.puzzle_goal_label
        )
        scramble_clicked = st.button("🎲 Scramble / Apply Board", width="stretch")

        if p_params_changed or scramble_clicked:
            st.session_state.puzzle_size_label = puzzle_size_label
            st.session_state.puzzle_preset = preset_puzzle
            st.session_state.puzzle_goal_label = goal_choice
            st.session_state.puzzle_goal = selected_goal

            if preset_puzzle == "Random Solvable Board":
                st.session_state.puzzle_state = EightPuzzle.generate_random_solvable(
                    goal_state=selected_goal, scramble_moves=scramble_depth
                )
            elif not is_15 and "Standard" in goal_choice:
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
                    goal_state=selected_goal, scramble_moves=depth_map[preset_puzzle]
                )

            st.session_state.puzzle_results = {}
            st.session_state.trigger_puzzle_anim = False
            st.session_state.puzzle_instant_mode = False
            if scramble_clicked:
                st.rerun()

        with st.expander("✏️ Enter Custom Board Tiles (Optional)", expanded=False):
            expected_n = 16 if is_15 else 9
            default_txt = ",".join(str(x) for x in st.session_state.puzzle_state)
            custom_str = st.text_input(
                f"Comma-separated {expected_n} tiles (0 = blank):",
                value=default_txt,
                key="custom_puzzle_txt",
            )
            if st.button("Apply Custom Board", width="stretch"):
                try:
                    parsed = [int(t.strip()) for t in custom_str.split(",") if t.strip() != ""]
                    if len(parsed) != expected_n or set(parsed) != set(range(expected_n)):
                        st.error(f"Provide exact {expected_n} unique integers 0..{expected_n - 1}.")
                    else:
                        st.session_state.puzzle_state = tuple(parsed)
                        st.session_state.puzzle_results = {}
                        st.session_state.trigger_puzzle_anim = False
                        st.session_state.puzzle_instant_mode = False
                        st.rerun()
                except ValueError:
                    st.error("Invalid integers. Example: 2,1,3,4,5,6,7,8,0")

        inv_count = EightPuzzle.count_inversions(st.session_state.puzzle_state)
        is_solv = EightPuzzle.is_solvable(
            st.session_state.puzzle_state, st.session_state.puzzle_goal
        )
        blank_row_txt = (
            f" | BlankRow={st.session_state.puzzle_state.index(0) // 4}"
            if is_15
            else ""
        )
        st.markdown(
            f"""
            <div class="metric-pill">
                🔍 <b>Inversions:</b> {inv_count}{blank_row_txt} |
                Status: <b>{'🟢 Solvable' if is_solv else '🔴 Unsolvable (Parity Mismatch)'}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### ⚡ Solver & Race Setup")
        p_c_strat, p_c_heur = st.columns([1.35, 1])
        with p_c_strat:
            p_solver_choice = st.selectbox(
                "Search Strategy:",
                [
                    "🏁 Race All 4 (A*, BFS, Greedy, Bidir BFS)",
                    "A* Search",
                    "Bidirectional BFS",
                    "Greedy Best-First",
                    "UCS (Dijkstra)",
                    "Breadth-First Search (BFS)",
                    "Depth-First Search (DFS)",
                ],
                index=0,
            )
        with p_c_heur:
            uses_p_heur = any(
                k in p_solver_choice for k in ("A*", "Greedy", "Race")
            )
            p_metric = st.selectbox(
                "Heuristic:",
                ["Manhattan", "Euclidean"],
                disabled=not uses_p_heur,
            )
            metric_key = p_metric.lower()

        uses_p_w = any(k in p_solver_choice for k in ("A*", "Race"))
        p_weight = st.slider(
            "Weighted A* Factor w (f = g + w·h):",
            min_value=1.0,
            max_value=3.0,
            value=1.0,
            step=0.25,
            disabled=not uses_p_w,
            key="p_weight_slider",
        )

        puzzle_solve_mode = st.radio(
            "Execution Mode:",
            ["🎬 Animated (Tiles Slide 1-by-1)", "⚡ Instant Solution (No Animation)"],
            horizontal=True,
            key="p_solve_mode_radio",
        )

        if st.button("🚀 Solve Puzzle", type="primary", width="stretch", disabled=not is_solv):
            env_puzzle = EightPuzzle(
                start_state=st.session_state.puzzle_state,
                goal_state=st.session_state.puzzle_goal,
            )
            pw_tag = f" (w={p_weight:.2g})" if p_weight > 1.0 else ""

            if p_solver_choice.startswith("🏁 Race"):
                agents = [
                    AStarAgent(
                        name=f"A*{pw_tag}",
                        heuristic_metric=metric_key,
                        weight=p_weight,
                    ),
                    BFSAgent(name="BFS"),
                    GreedyAgent(name="Greedy", heuristic_metric=metric_key),
                    BidirectionalBFSAgent(name="Bidir BFS"),
                ]
                is_p_race = True
            else:
                agent_dict = {
                    "A* Search": AStarAgent(
                        name=f"A* ({p_metric}){pw_tag}",
                        heuristic_metric=metric_key,
                        weight=p_weight,
                    ),
                    "Bidirectional BFS": BidirectionalBFSAgent(name="Bidirectional BFS"),
                    "Greedy Best-First": GreedyAgent(
                        name=f"Greedy ({p_metric})", heuristic_metric=metric_key
                    ),
                    "UCS (Dijkstra)": UCSAgent(name="UCS (Dijkstra)"),
                    "Breadth-First Search (BFS)": BFSAgent(name="BFS"),
                    "Depth-First Search (DFS)": DFSAgent(name="DFS"),
                }
                agents = [agent_dict[p_solver_choice]]
                is_p_race = False

            p_results = {}
            with st.spinner("Solving puzzle..."):
                for ag in agents:
                    p_results[ag.name] = ag.solve(env_puzzle)
            st.session_state.puzzle_results = p_results
            st.session_state.puzzle_race_mode = is_p_race
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

        puzzle_payload = build_puzzle_payload(
            start_state=st.session_state.puzzle_state,
            goal_state=st.session_state.puzzle_goal,
            results=st.session_state.puzzle_results,
            autoplay=p_autoplay_now,
            instant=p_instant_now,
            race_mode=st.session_state.puzzle_race_mode,
            canvas_px=440,
            run_id=st.session_state.puzzle_solve_seq,
        )
        p_comp_res = puzzle_component(
            data=puzzle_payload,
            on_puzzle_edit_change=lambda: None,
            key=f"puzzle_interactive_canvas_{_PUZZLE_VER}",
        )
        p_edit = getattr(p_comp_res, "puzzle_edit", None)
        if p_edit and "board" in p_edit:
            st.session_state.puzzle_state = tuple(int(x) for x in p_edit["board"])
            st.session_state.puzzle_results = {}
            st.session_state.trigger_puzzle_anim = False
            st.session_state.puzzle_instant_mode = False
            st.rerun()

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
                    p_dim = 4 if len(st.session_state.puzzle_state) == 16 else 3
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
                        r0, c0 = divmod(curr_zero, p_dim)
                        r1, c1 = divmod(prev_zero, p_dim)
                        dr, dc = r1 - r0, c1 - c0
                        direction = (
                            "UP"
                            if dr == -1
                            else ("DOWN" if dr == 1 else ("LEFT" if dc == -1 else "RIGHT"))
                        )
                        dir_lbl, dir_clr = dir_styles[direction]
                        board_str = " | ".join(
                            " ".join(str(x) if x != 0 else "·" for x in curr_b[r * p_dim : (r + 1) * p_dim])
                            for r in range(p_dim)
                        )
                        goal_tag = (
                            ' <span style="color:#50fa7b; font-weight:700; margin-left:6px;">🎉 Goal</span>'
                            if idx == total_moves
                            else ""
                        )
                        rows_html.append(
                            f'<div style="display:grid; grid-template-columns:56px 78px 92px 1fr 220px; align-items:center; gap:8px; padding:6px 10px; border-bottom:1px solid #343746; font-size:0.84rem;">'
                            f'<span style="color:#bd93f9; font-weight:700; font-family:monospace;">#{idx:02d}</span>'
                            f'<span><span style="background:#ff79c6; color:#282a36; font-weight:800; padding:2px 8px; border-radius:4px;">Tile {moved_t}</span></span>'
                            f'<span style="color:{dir_clr}; font-weight:700;">{dir_lbl}</span>'
                            f'<span style="color:#f8f8f2;">({r0},{c0}) → ({r1},{c1}){goal_tag}</span>'
                            f'<span style="color:#8be9fd; font-family:monospace; font-size:0.75rem; background:#282a36; padding:2px 6px; border-radius:4px; text-align:center;">{board_str}</span>'
                            f'</div>'
                        )

                    trunc_note = (
                        f'<div style="padding:6px 10px; color:#ffb86c; font-size:0.8rem;">Showing first {max_display} of {total_moves} total moves.</div>'
                        if show_truncated
                        else ""
                    )
                    log_html = (
                        '<div style="border:1px solid #44475a; border-radius:8px; background:#21222c; overflow:hidden; margin-bottom:6px;">'
                        '<div style="display:grid; grid-template-columns:56px 78px 92px 1fr 220px; gap:8px; padding:7px 10px; background:#282a36; border-bottom:1px solid #6272a4; font-size:0.78rem; font-weight:700; color:#8be9fd; text-transform:uppercase;">'
                        '<span>Step</span><span>Tile</span><span>Direction</span><span>Move Coordinates</span><span style="text-align:center;">Board Snapshot</span>'
                        '</div>'
                        f'<div style="max-height:260px; overflow-y:auto;">{"".join(rows_html)}{trunc_note}</div>'
                        '</div>'
                    )
                    st.html(log_html)

        if len(st.session_state.puzzle_results) > 1:
            with st.expander("📊 View Puzzle Multi-Algorithm Comparison Table", expanded=False):
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
            <span style="font-size:0.8rem; color:#8be9fd; font-weight:600;">Classical State-Space Search · Heuristics · Solvability Proofs</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.html(
        """
        <div style="display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:10px; margin-bottom:10px;">
            <div style="background:#21222c; border:1px solid #44475a; border-left:4px solid #bd93f9; border-radius:8px; padding:10px 12px;">
                <div style="font-size:0.72rem; color:#6272a4; text-transform:uppercase; font-weight:700;">Search Strategies</div>
                <div style="font-size:1.25rem; font-weight:800; color:#f8f8f2; margin:2px 0;">6 Solvers</div>
                <div style="font-size:0.76rem; color:#bd93f9;">BFS · DFS · UCS · Greedy · A* · Bidir</div>
            </div>
            <div style="background:#21222c; border:1px solid #44475a; border-left:4px solid #8be9fd; border-radius:8px; padding:10px 12px;">
                <div style="font-size:0.72rem; color:#6272a4; text-transform:uppercase; font-weight:700;">Admissible Heuristics</div>
                <div style="font-size:1.25rem; font-weight:800; color:#f8f8f2; margin:2px 0;">L₁ &amp; L₂</div>
                <div style="font-size:0.76rem; color:#8be9fd;">Manhattan &amp; Euclidean Metrics</div>
            </div>
            <div style="background:#21222c; border:1px solid #44475a; border-left:4px solid #50fa7b; border-radius:8px; padding:10px 12px;">
                <div style="font-size:0.72rem; color:#6272a4; text-transform:uppercase; font-weight:700;">Optimal Guarantees</div>
                <div style="font-size:1.25rem; font-weight:800; color:#f8f8f2; margin:2px 0;">4 Optimal</div>
                <div style="font-size:0.76rem; color:#50fa7b;">BFS · UCS · A* (w=1) · Bidir BFS</div>
            </div>
            <div style="background:#21222c; border:1px solid #44475a; border-left:4px solid #ff79c6; border-radius:8px; padding:10px 12px;">
                <div style="font-size:0.72rem; color:#6272a4; text-transform:uppercase; font-weight:700;">State Guarantees</div>
                <div style="font-size:1.25rem; font-weight:800; color:#f8f8f2; margin:2px 0;">100% Solvable</div>
                <div style="font-size:0.76rem; color:#ff79c6;">Carved Mazes &amp; Parity Puzzles</div>
            </div>
        </div>
        """
    )

    tab_matrix, tab_heuristics, tab_solvability = st.tabs(
        [
            "📊 Complexity Matrix & Algorithm Cards",
            "🧭 Heuristics & Weighted A*",
            "🧩 Puzzle Parity & Maze Guarantees",
        ]
    )

    with tab_matrix:
        st.html(
            """
            <div style="border:1px solid #44475a; border-radius:10px; background:#21222c; overflow:hidden; margin-bottom:12px;">
                <div style="padding:8px 14px; background:#282a36; border-bottom:1px solid #44475a; display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; color:#f8f8f2; font-size:0.9rem;">Comparative Asymptotic Complexity</span>
                    <span style="font-size:0.75rem; color:#6272a4;">b = branching factor · d = optimal depth · m = max depth · C*/ε = min-cost depth</span>
                </div>
                <div style="overflow-x:auto;">
                    <table style="width:100%; border-collapse:collapse; font-size:0.84rem; color:#f8f8f2;">
                        <thead>
                            <tr style="background:#262835; color:#8be9fd; text-align:left; font-size:0.75rem; text-transform:uppercase; border-bottom:1px solid #44475a;">
                                <th style="padding:9px 12px;">Algorithm</th>
                                <th style="padding:9px 12px;">Category</th>
                                <th style="padding:9px 12px;">Priority / Frontier</th>
                                <th style="padding:9px 12px;">Time</th>
                                <th style="padding:9px 12px;">Space</th>
                                <th style="padding:9px 12px;">Complete?</th>
                                <th style="padding:9px 12px;">Optimal?</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom:1px solid #343746;">
                                <td style="padding:9px 12px; font-weight:700; color:#bd93f9;">Breadth-First Search (BFS)</td>
                                <td style="padding:9px 12px;"><span style="background:#282a36; border:1px solid #8be9fd; color:#8be9fd; padding:2px 8px; border-radius:999px; font-size:0.73rem;">Uninformed</span></td>
                                <td style="padding:9px 12px; font-family:monospace;">FIFO Queue · g(n) = depth</td>
                                <td style="padding:9px 12px; font-family:monospace; color:#f1fa8c;">O(b<sup>d</sup>)</td>
                                <td style="padding:9px 12px; font-family:monospace; color:#f1fa8c;">O(b<sup>d</sup>)</td>
                                <td style="padding:9px 12px;"><span style="color:#50fa7b; font-weight:700;">Yes</span></td>
                                <td style="padding:9px 12px;"><span style="color:#50fa7b; font-weight:700;">Yes</span> <span style="color:#6272a4; font-size:0.76rem;">(unit cost)</span></td>
                            </tr>
                            <tr style="border-bottom:1px solid #343746;">
                                <td style="padding:9px 12px; font-weight:700; color:#ffb86c;">Depth-First Search (DFS)</td>
                                <td style="padding:9px 12px;"><span style="background:#282a36; border:1px solid #8be9fd; color:#8be9fd; padding:2px 8px; border-radius:999px; font-size:0.73rem;">Uninformed</span></td>
                                <td style="padding:9px 12px; font-family:monospace;">LIFO Stack · visited set</td>
                                <td style="padding:9px 12px; font-family:monospace; color:#ffb86c;">O(b<sup>m</sup>)</td>
                                <td style="padding:9px 12px; font-family:monospace; color:#50fa7b;">O(b · m)</td>
                                <td style="padding:9px 12px;"><span style="color:#50fa7b; font-weight:700;">Yes</span> <span style="color:#6272a4; font-size:0.76rem;">(finite graph)</span></td>
                                <td style="padding:9px 12px;"><span style="color:#ff5555; font-weight:700;">No</span></td>
                            </tr>
                            <tr style="border-bottom:1px solid #343746;">
                                <td style="padding:9px 12px; font-weight:700; color:#8be9fd;">Uniform Cost Search (UCS)</td>
                                <td style="padding:9px 12px;"><span style="background:#282a36; border:1px solid #8be9fd; color:#8be9fd; padding:2px 8px; border-radius:999px; font-size:0.73rem;">Uninformed</span></td>
                                <td style="padding:9px 12px; font-family:monospace;">Min-Heap · f(n) = g(n)</td>
                                <td style="padding:9px 12px; font-family:monospace; color:#f1fa8c;">O(b<sup>1 + ⌊C*/ε⌋</sup>)</td>
                                <td style="padding:9px 12px; font-family:monospace; color:#f1fa8c;">O(b<sup>1 + ⌊C*/ε⌋</sup>)</td>
                                <td style="padding:9px 12px;"><span style="color:#50fa7b; font-weight:700;">Yes</span></td>
                                <td style="padding:9px 12px;"><span style="color:#50fa7b; font-weight:700;">Yes</span></td>
                            </tr>
                            <tr style="border-bottom:1px solid #343746;">
                                <td style="padding:9px 12px; font-weight:700; color:#ff79c6;">Greedy Best-First Search</td>
                                <td style="padding:9px 12px;"><span style="background:#282a36; border:1px solid #ff79c6; color:#ff79c6; padding:2px 8px; border-radius:999px; font-size:0.73rem;">Informed</span></td>
                                <td style="padding:9px 12px; font-family:monospace;">Min-Heap · f(n) = h(n)</td>
                                <td style="padding:9px 12px; font-family:monospace; color:#ffb86c;">O(b<sup>m</sup>)</td>
                                <td style="padding:9px 12px; font-family:monospace; color:#ffb86c;">O(b<sup>m</sup>)</td>
                                <td style="padding:9px 12px;"><span style="color:#50fa7b; font-weight:700;">Yes</span> <span style="color:#6272a4; font-size:0.76rem;">(graph search)</span></td>
                                <td style="padding:9px 12px;"><span style="color:#ff5555; font-weight:700;">No</span></td>
                            </tr>
                            <tr style="border-bottom:1px solid #343746;">
                                <td style="padding:9px 12px; font-weight:700; color:#50fa7b;">A* / Weighted A*</td>
                                <td style="padding:9px 12px;"><span style="background:#282a36; border:1px solid #ff79c6; color:#ff79c6; padding:2px 8px; border-radius:999px; font-size:0.73rem;">Informed</span></td>
                                <td style="padding:9px 12px; font-family:monospace;">Min-Heap · f(n) = g(n) + w·h(n)</td>
                                <td style="padding:9px 12px; font-family:monospace; color:#50fa7b;">Optimally efficient*</td>
                                <td style="padding:9px 12px; font-family:monospace; color:#f1fa8c;">O(b<sup>d</sup>)</td>
                                <td style="padding:9px 12px;"><span style="color:#50fa7b; font-weight:700;">Yes</span></td>
                                <td style="padding:9px 12px;"><span style="color:#f1fa8c; font-weight:700;">w = 1 &amp; admissible h</span></td>
                            </tr>
                            <tr>
                                <td style="padding:9px 12px; font-weight:700; color:#f1fa8c;">Bidirectional BFS</td>
                                <td style="padding:9px 12px;"><span style="background:#282a36; border:1px solid #bd93f9; color:#bd93f9; padding:2px 8px; border-radius:999px; font-size:0.73rem;">Meet-in-Middle</span></td>
                                <td style="padding:9px 12px; font-family:monospace;">Dual FIFO Queues (S→ &amp; ←G)</td>
                                <td style="padding:9px 12px; font-family:monospace; color:#50fa7b;">O(b<sup>d/2</sup>)</td>
                                <td style="padding:9px 12px; font-family:monospace; color:#50fa7b;">O(b<sup>d/2</sup>)</td>
                                <td style="padding:9px 12px;"><span style="color:#50fa7b; font-weight:700;">Yes</span></td>
                                <td style="padding:9px 12px;"><span style="color:#50fa7b; font-weight:700;">Yes</span> <span style="color:#6272a4; font-size:0.76rem;">(unit cost)</span></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:10px;">
                <div style="background:#21222c; border:1px solid #44475a; border-top:3px solid #bd93f9; border-radius:8px; padding:11px 13px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-weight:800; color:#bd93f9; font-size:0.9rem;">BFS</span>
                        <span style="font-size:0.72rem; color:#8be9fd; font-family:monospace;">FIFO Queue</span>
                    </div>
                    <div style="font-size:0.8rem; color:#f8f8f2; line-height:1.45;">Expands states strictly level-by-level in waves. Always finds the shortest path on unweighted grids and sliding puzzles, at the cost of storing the full frontier.</div>
                </div>
                <div style="background:#21222c; border:1px solid #44475a; border-top:3px solid #ffb86c; border-radius:8px; padding:11px 13px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-weight:800; color:#ffb86c; font-size:0.9rem;">DFS</span>
                        <span style="font-size:0.72rem; color:#8be9fd; font-family:monospace;">LIFO Stack</span>
                    </div>
                    <div style="font-size:0.8rem; color:#f8f8f2; line-height:1.45;">Plunges deeply along one branch before backtracking. Uses minimal linear stack memory O(b·m), but paths often wind far longer than optimal.</div>
                </div>
                <div style="background:#21222c; border:1px solid #44475a; border-top:3px solid #8be9fd; border-radius:8px; padding:11px 13px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-weight:800; color:#8be9fd; font-size:0.9rem;">Uniform Cost (UCS)</span>
                        <span style="font-size:0.72rem; color:#8be9fd; font-family:monospace;">Min-Heap g(n)</span>
                    </div>
                    <div style="font-size:0.8rem; color:#f8f8f2; line-height:1.45;">Dijkstra's algorithm generalized to state-space graphs. Expands lowest cumulative cost g(n) first without any goal heuristic (h = 0).</div>
                </div>
                <div style="background:#21222c; border:1px solid #44475a; border-top:3px solid #ff79c6; border-radius:8px; padding:11px 13px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-weight:800; color:#ff79c6; font-size:0.9rem;">Greedy Best-First</span>
                        <span style="font-size:0.72rem; color:#8be9fd; font-family:monospace;">Min-Heap h(n)</span>
                    </div>
                    <div style="font-size:0.8rem; color:#f8f8f2; line-height:1.45;">Rushes straight toward the goal estimate h(n) ignoring g(n). Very fast in open space, but easily lured into dead-end U-traps and suboptimal detours.</div>
                </div>
                <div style="background:#21222c; border:1px solid #44475a; border-top:3px solid #50fa7b; border-radius:8px; padding:11px 13px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-weight:800; color:#50fa7b; font-size:0.9rem;">A* &amp; Weighted A*</span>
                        <span style="font-size:0.72rem; color:#8be9fd; font-family:monospace;">g(n) + w·h(n)</span>
                    </div>
                    <div style="font-size:0.8rem; color:#f8f8f2; line-height:1.45;">Balances exact past cost g(n) with admissible future estimate h(n). Optimal at w=1; raising w &gt; 1 focuses the beam to trade optimality for speed.</div>
                </div>
                <div style="background:#21222c; border:1px solid #44475a; border-top:3px solid #f1fa8c; border-radius:8px; padding:11px 13px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-weight:800; color:#f1fa8c; font-size:0.9rem;">Bidirectional BFS</span>
                        <span style="font-size:0.72rem; color:#8be9fd; font-family:monospace;">S→ &amp; ←G Waves</span>
                    </div>
                    <div style="font-size:0.8rem; color:#f8f8f2; line-height:1.45;">Runs two concurrent BFS frontiers from Start and Goal until they collide, cutting search volume from b<sup>d</sup> down to 2·b<sup>d/2</sup>.</div>
                </div>
            </div>
            """
        )

    with tab_heuristics:
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            with st.container(border=True):
                st.markdown("##### 📐 Admissible Distance Heuristics")
                st.latex(r"h_{\text{Manhattan}}(n) = |r_n - r_{\text{goal}}| + |c_n - c_{\text{goal}}|")
                st.latex(r"h_{\text{Euclidean}}(n) = \sqrt{(r_n - r_{\text{goal}})^2 + (c_n - c_{\text{goal}})^2}")
                st.markdown(
                    r"""
                    * **Admissibility ($h(n) \le h^*(n)$):** Never overestimates true remaining cost, preserving A\*'s shortest-path guarantee at $w = 1$.
                    * **Dominance ($L_2 \le L_1 \le h^*$):** On 4-way grids and sliding puzzles, **Manhattan ($L_1$)** equals the exact obstacle-free distance and is tighter than Euclidean ($L_2$), pruning more dead branches.
                    """
                )
        with col_h2:
            with st.container(border=True):
                st.markdown(r"##### ⚖️ Weighted A\* Tradeoff Spectrum")
                st.latex(r"f(n) = g(n) + w \cdot h(n), \quad w \in [1.0, 3.0]")
                st.html(
                    """
                    <div style="display:flex; flex-direction:column; gap:7px; margin:6px 0 8px 0; font-size:0.81rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; background:#21222c; padding:6px 10px; border-radius:6px; border-left:3px solid #8be9fd;">
                            <span><b>w = 0</b> · Pure Uniform Cost</span>
                            <span style="color:#8be9fd;">Optimal · Broadest wave</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; background:#21222c; padding:6px 10px; border-radius:6px; border-left:3px solid #50fa7b;">
                            <span><b>w = 1.0</b> · Standard A*</span>
                            <span style="color:#50fa7b;">Optimal · Minimum optimal nodes</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; background:#21222c; padding:6px 10px; border-radius:6px; border-left:3px solid #ff79c6;">
                            <span><b>w &gt; 1.0</b> · Weighted A*</span>
                            <span style="color:#ff79c6;">Cost ≤ w·C* · Faster convergence</span>
                        </div>
                    </div>
                    """
                )

        with st.container(border=True):
            st.markdown("##### ⚡ Exponential Savings: Unidirectional `O(b^d)` vs. Bidirectional `O(2·b^(d/2))`")
            c_b, c_d = st.columns(2)
            with c_b:
                sim_b = st.slider("Branching Factor (b):", min_value=2, max_value=4, value=3, key="theory_sim_b")
            with c_d:
                sim_d = st.slider("Solution Depth (d):", min_value=4, max_value=18, value=12, step=2, key="theory_sim_d")

            uni_nodes = sim_b ** sim_d
            bi_nodes = 2 * (sim_b ** (sim_d // 2))
            speedup = uni_nodes / bi_nodes
            bi_pct = max(2.0, min(100.0, (bi_nodes / uni_nodes) * 100.0))

            st.html(
                f"""
                <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-top:4px; margin-bottom:6px;">
                    <div style="background:#21222c; border:1px solid #44475a; border-radius:8px; padding:8px 12px;">
                        <div style="font-size:0.74rem; color:#bd93f9; font-weight:700;">BFS / Standard Wave (b<sup>d</sup>)</div>
                        <div style="font-size:1.15rem; font-weight:800; color:#f8f8f2; font-family:monospace;">{uni_nodes:,} states</div>
                    </div>
                    <div style="background:#21222c; border:1px solid #44475a; border-radius:8px; padding:8px 12px;">
                        <div style="font-size:0.74rem; color:#50fa7b; font-weight:700;">Bidirectional BFS (2·b<sup>d/2</sup>)</div>
                        <div style="font-size:1.15rem; font-weight:800; color:#50fa7b; font-family:monospace;">{bi_nodes:,} states</div>
                    </div>
                    <div style="background:#21222c; border:1px solid #44475a; border-radius:8px; padding:8px 12px;">
                        <div style="font-size:0.74rem; color:#f1fa8c; font-weight:700;">Theoretical Pruning Factor</div>
                        <div style="font-size:1.15rem; font-weight:800; color:#f1fa8c; font-family:monospace;">{speedup:,.1f}× fewer</div>
                    </div>
                </div>
                <div style="background:#21222c; border:1px solid #44475a; border-radius:6px; padding:8px 12px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#6272a4; margin-bottom:4px;">
                        <span>Relative State-Space Footprint</span>
                        <span>Bidirectional = {bi_pct:.3f}% of Unidirectional</span>
                    </div>
                    <div style="height:10px; background:#44475a; border-radius:999px; overflow:hidden; margin-bottom:4px;">
                        <div style="width:100%; height:100%; background:#bd93f9;"></div>
                    </div>
                    <div style="height:10px; background:#44475a; border-radius:999px; overflow:hidden;">
                        <div style="width:{bi_pct:.2f}%; height:100%; background:#50fa7b;"></div>
                    </div>
                </div>
                """
            )

    with tab_solvability:
        col_s1, col_s2 = st.columns([1.15, 0.85])
        with col_s1:
            with st.container(border=True):
                st.markdown("##### 🔢 General $W \\times W$ Sliding Puzzle Solvability Theorem")
                st.latex(r"\mathrm{Inv}(S) = \sum_{0 \le i < j < W^2} [\,S_i > S_j \land S_i \ne 0 \land S_j \ne 0\,]")
                st.markdown(
                    r"""
                    * **Odd Width ($3\times 3$ 8-Puzzle):** Horizontal moves keep $\mathrm{Inv}(S)$ unchanged and vertical moves shift a tile past $W - 1 = 2$ neighbors (even), so parity never changes:
                      $$\mathrm{Inv}(S) \equiv \mathrm{Inv}(G) \pmod 2$$
                    * **Even Width ($4\times 4$ 15-Puzzle):** Vertical blank moves skip $W - 1 = 3$ tiles (odd), flipping both $\mathrm{Inv}(S)$ and $r_{\text{blank}}$ parity together so their sum stays invariant:
                      $$\bigl(\mathrm{Inv}(S) + r_{\text{blank}}(S)\bigr) \equiv \bigl(\mathrm{Inv}(G) + r_{\text{blank}}(G)\bigr) \pmod 2$$
                    """
                )
                st.html(
                    """
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:6px;">
                        <div style="background:#21222c; border:1px solid #50fa7b; border-radius:8px; padding:8px 10px;">
                            <div style="font-size:0.78rem; font-weight:700; color:#50fa7b; margin-bottom:4px;">✅ Solvable (Inv = 2, Even)</div>
                            <div style="font-family:monospace; font-size:0.8rem; color:#f8f8f2;">1 2 3 &nbsp;|&nbsp; 4 · <span style="color:#50fa7b; font-weight:700;">6</span> &nbsp;|&nbsp; <span style="color:#50fa7b; font-weight:700;">7 5</span> 8</div>
                            <div style="font-size:0.73rem; color:#6272a4; margin-top:3px;">Two inversions (6 &gt; 5, 7 &gt; 5) preserve even parity.</div>
                        </div>
                        <div style="background:#21222c; border:1px solid #ff5555; border-radius:8px; padding:8px 10px;">
                            <div style="font-size:0.78rem; font-weight:700; color:#ff5555; margin-bottom:4px;">❌ Unsolvable (Inv = 1, Odd)</div>
                            <div style="font-family:monospace; font-size:0.8rem; color:#f8f8f2;"><span style="color:#ff5555; font-weight:700;">2 1</span> 3 &nbsp;|&nbsp; 4 5 6 &nbsp;|&nbsp; 7 8 ·</div>
                            <div style="font-size:0.73rem; color:#6272a4; margin-top:3px;">Single transposition (2 &gt; 1) flips parity forever.</div>
                        </div>
                    </div>
                    """
                )
        with col_s2:
            with st.container(border=True):
                st.markdown("##### 🗺️ Guaranteed-Solvable Maze Architecture")
                st.html(
                    """
                    <div style="display:flex; flex-direction:column; gap:8px; font-size:0.82rem; color:#f8f8f2;">
                        <div style="background:#21222c; border:1px solid #44475a; border-left:3px solid #bd93f9; border-radius:6px; padding:8px 10px;">
                            <div style="font-weight:700; color:#bd93f9;">1. Spanning-Tree Labyrinths</div>
                            <div style="color:#f8f8f2; margin-top:2px;"><b>Recursive Backtracker</b> &amp; <b>Prim's Perfect Maze</b> carve connected odd-coordinate corridors with zero isolated rooms.</div>
                        </div>
                        <div style="background:#21222c; border:1px solid #44475a; border-left:3px solid #8be9fd; border-radius:6px; padding:8px 10px;">
                            <div style="font-weight:700; color:#8be9fd;">2. High-Density Verification</div>
                            <div style="color:#f8f8f2; margin-top:2px;">Random obstacle grids (up to <b>35×35</b> at <b>0.50</b> density) are immediately tested with a fast BFS reachability pass.</div>
                        </div>
                        <div style="background:#21222c; border:1px solid #44475a; border-left:3px solid #50fa7b; border-radius:6px; padding:8px 10px;">
                            <div style="font-weight:700; color:#50fa7b;">3. Shortest-Carve Fallback</div>
                            <div style="color:#f8f8f2; margin-top:2px;">If blocked, MazeMind AI runs a randomized 0-1 minimum-wall pathfinder and opens the minimal bottleneck walls needed to guarantee Start → Goal connectivity.</div>
                        </div>
                    </div>
                    """
                )
