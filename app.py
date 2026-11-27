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
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### 1. Comparative Complexity Matrix")
    theory_data = [
        {
            "Algorithm": "Breadth-First Search (BFS)",
            "Type": "Uninformed",
            "Evaluation": "g(n) = depth(n)",
            "Time": "O(b^d)",
            "Space": "O(b^d)",
            "Optimal?": "Yes (unit cost)",
        },
        {
            "Algorithm": "Depth-First Search (DFS)",
            "Type": "Uninformed",
            "Evaluation": "LIFO stack order",
            "Time": "O(b^m)",
            "Space": "O(b · m)",
            "Optimal?": "No",
        },
        {
            "Algorithm": "Uniform Cost Search (UCS)",
            "Type": "Uninformed",
            "Evaluation": "g(n) = path cost",
            "Time": "O(b^(C*/ε))",
            "Space": "O(b^(C*/ε))",
            "Optimal?": "Yes",
        },
        {
            "Algorithm": "Greedy Best-First Search",
            "Type": "Informed",
            "Evaluation": "f(n) = h(n)",
            "Time": "O(b^m)",
            "Space": "O(b^m)",
            "Optimal?": "No",
        },
        {
            "Algorithm": "A* / Weighted A*",
            "Type": "Informed",
            "Evaluation": "f(n) = g(n) + w·h(n)",
            "Time": "Optimal at w=1",
            "Space": "O(b^d)",
            "Optimal?": "Yes when w=1 & h admissible",
        },
        {
            "Algorithm": "Bidirectional BFS",
            "Type": "Meet-in-the-Middle",
            "Evaluation": "S→ & ←G concurrently",
            "Time": "O(b^(d/2))",
            "Space": "O(b^(d/2))",
            "Optimal?": "Yes (unit cost)",
        },
    ]
    st.dataframe(pd.DataFrame(theory_data), width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("##### 2. Admissible Heuristics & Weighted A*")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown(
            r"""
            **Manhattan Distance ($L_1$ Norm) & Euclidean ($L_2$ Norm):**
            $$h_{\text{Manhattan}}(n) = |r_n - r_{\text{goal}}| + |c_n - c_{\text{goal}}|$$
            $$h_{\text{Euclidean}}(n) = \sqrt{(r_n - r_{\text{goal}})^2 + (c_n - c_{\text{goal}})^2}$$
            """
        )
    with col_h2:
        st.markdown(
            r"""
            **Weighted A\* Tradeoff ($f(n) = g(n) + w \cdot h(n)$):**
            * At $w = 1.0$, A\* guarantees shortest optimal paths.
            * Increasing $w > 1.0$ trades strict optimality for faster frontier convergence.
            """
        )

    st.markdown("---")
    st.markdown("##### 3. General $W \\times W$ Sliding Puzzle Solvability Theorem")
    st.markdown(
        r"""
        $$\mathrm{Inv}(S) = \sum_{1 \le i < j < W^2} [S_i > S_j \land S_i \ne 0 \land S_j \ne 0]$$
        * **Odd Width ($3\times 3$ 8-Puzzle):** $\mathrm{Inv}(S) \equiv \mathrm{Inv}(G) \pmod 2$.
        * **Even Width ($4\times 4$ 15-Puzzle):** Vertical blank slides shift $\mathrm{Inv}$ parity, so the invariant pairs inversions with the blank tile's row:
          $$(\mathrm{Inv}(S) + r_{\text{blank}}(S)) \equiv (\mathrm{Inv}(G) + r_{\text{blank}}(G)) \pmod 2$$
        """
    )
