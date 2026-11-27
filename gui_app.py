"""
MazeMind AI — Native Desktop GUI Application (Dracula Theme).
Specifications: https://draculatheme.com/spec

Features:
- Dual Problem Modes: 2D Grid Maze (Random & Labyrinths) & 8-Puzzle / 15-Puzzle
- Algorithms: BFS, DFS, UCS (Dijkstra), Greedy, A* & Weighted A*, Bidirectional BFS
- Tile-by-Tile Animation + Bidirectional Dual-Color Highlights
- Replay uses Already-Solved Data Without Re-Solving; Live Speed Slider Mid-Playback
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Tuple

import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from mazemind.environments import GridMaze, EightPuzzle
from mazemind.search_agents import (
    SearchResult,
    BFSAgent,
    DFSAgent,
    UCSAgent,
    GreedyAgent,
    AStarAgent,
    BidirectionalBFSAgent,
)

# Official Dracula Theme Palette (https://draculatheme.com/spec)
DRACULA = {
    "bg": "#282a36",
    "current_line": "#44475a",
    "selection": "#44475a",
    "fg": "#f8f8f2",
    "comment": "#6272a4",
    "cyan": "#8be9fd",
    "bwd_blue": "#38bdf8",
    "green": "#50fa7b",
    "orange": "#ffb86c",
    "pink": "#ff79c6",
    "purple": "#bd93f9",
    "red": "#ff5555",
    "yellow": "#f1fa8c",
    "dark_wall": "#191a21",
}


class MazeDesktopApp:
    """Native Desktop GUI for MazeMind AI with Dracula Theme."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MazeMind AI — Search & Puzzle Visualizer")
        self.root.geometry("1180x810")
        self.root.minsize(980, 690)
        self.root.configure(bg=DRACULA["bg"])

        # ---------------- Maze State ----------------
        self.rows = 15
        self.cols = 15
        self.density = 0.22
        self.cell_size = 32
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        init_m = GridMaze.generate_solvable(self.rows, self.cols, density=self.density, seed=42)
        self.grid = init_m.grid

        self.maze_animating = False
        self.maze_paused = False
        self.maze_after_id: Optional[str] = None
        self.maze_step = 0
        self.maze_path_step = 0
        self.maze_history: List[Tuple[int, int]] = []
        self.maze_sides: Optional[List[int]] = None
        self.maze_path: List[Tuple[int, int]] = []
        self.maze_result: Optional[SearchResult] = None

        # ---------------- Puzzle State ----------------
        self.puzzle_dim = 3
        self.puzzle_goal = EightPuzzle.STANDARD_GOAL
        self.puzzle_state = EightPuzzle.generate_random_solvable(
            goal_state=self.puzzle_goal, scramble_moves=18, seed=7
        )
        self.puzzle_animating = False
        self.puzzle_paused = False
        self.puzzle_after_id: Optional[str] = None
        self.puzzle_step = 0
        self.puzzle_path: List[Tuple[int, ...]] = []
        self.puzzle_moves_info: List[Tuple[Optional[int], str]] = []
        self.puzzle_result: Optional[SearchResult] = None

        self._setup_styles()
        self._build_ui()
        self._redraw_maze()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=DRACULA["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=DRACULA["current_line"],
            foreground=DRACULA["fg"],
            padding=[14, 6],
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", DRACULA["purple"])],
            foreground=[("selected", DRACULA["bg"])],
        )
        style.configure("TLabel", background=DRACULA["current_line"], foreground=DRACULA["fg"], font=("Segoe UI", 9))
        style.configure("TFrame", background=DRACULA["current_line"])
        style.configure("TCombobox", font=("Segoe UI", 9))

    def _build_ui(self):
        header = tk.Frame(self.root, bg=DRACULA["current_line"], padx=16, pady=9)
        header.pack(fill=tk.X)

        tk.Label(header, text="🧠 MazeMind AI", font=("Segoe UI", 16, "bold"), bg=DRACULA["current_line"], fg=DRACULA["purple"]).pack(side=tk.LEFT)
        tk.Label(header, text="Intelligent Puzzle Solver & Search Algorithm Visualizer", font=("Segoe UI", 10), bg=DRACULA["current_line"], fg=DRACULA["cyan"]).pack(side=tk.LEFT, padx=(12, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        self.tab_maze = tk.Frame(self.notebook, bg=DRACULA["bg"])
        self.notebook.add(self.tab_maze, text=" 🗺️ 2D Grid Maze Solver ")

        self.tab_puzzle = tk.Frame(self.notebook, bg=DRACULA["bg"])
        self.notebook.add(self.tab_puzzle, text=" 🧩 8-Puzzle / 15-Puzzle ")

        self._build_maze_tab()
        self._build_puzzle_tab()

    # =====================================================
    # TAB 1: 2D Grid Maze Implementation
    # =====================================================
    def _build_maze_tab(self):
        container = tk.Frame(self.tab_maze, bg=DRACULA["bg"])
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        sidebar = tk.Frame(container, bg=DRACULA["current_line"], width=330, padx=12, pady=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar.pack_propagate(False)

        viewport = tk.Frame(container, bg=DRACULA["bg"])
        viewport.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        size_dens_row = tk.Frame(sidebar, bg=DRACULA["current_line"])
        size_dens_row.pack(fill=tk.X, pady=(0, 4))

        sz_col = tk.Frame(size_dens_row, bg=DRACULA["current_line"])
        sz_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Label(sz_col, text="Grid Size (N×N):", font=("Segoe UI", 8, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"]).pack(anchor="w")
        self.maze_size_scale = tk.Scale(sz_col, from_=8, to=35, orient=tk.HORIZONTAL, bg=DRACULA["current_line"], fg=DRACULA["fg"], troughcolor=DRACULA["bg"], highlightthickness=0)
        self.maze_size_scale.set(15)
        self.maze_size_scale.pack(fill=tk.X)

        dn_col = tk.Frame(size_dens_row, bg=DRACULA["current_line"])
        dn_col.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))
        tk.Label(dn_col, text="Wall Density (%):", font=("Segoe UI", 8, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"]).pack(anchor="w")
        self.maze_density_scale = tk.Scale(dn_col, from_=5, to=50, orient=tk.HORIZONTAL, bg=DRACULA["current_line"], fg=DRACULA["fg"], troughcolor=DRACULA["bg"], highlightthickness=0)
        self.maze_density_scale.set(22)
        self.maze_density_scale.pack(fill=tk.X)

        tk.Label(sidebar, text="Maze Preset:", font=("Segoe UI", 8, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"]).pack(anchor="w")
        self.maze_preset_var = tk.StringVar(value="Random Obstacles")
        ttk.Combobox(
            sidebar,
            textvariable=self.maze_preset_var,
            values=[
                "Random Obstacles",
                "Recursive Backtracker (Labyrinth)",
                "Prim's Perfect Maze",
                "Open Room (No Walls)",
                "Center Wall Barrier",
                "Snake Labyrinth",
            ],
            state="readonly",
        ).pack(fill=tk.X, pady=(2, 6))

        m_row = tk.Frame(sidebar, bg=DRACULA["current_line"])
        m_row.pack(fill=tk.X, pady=(0, 6))
        tk.Button(m_row, text="🎲 New Solvable Maze", bg=DRACULA["comment"], fg="white", font=("Segoe UI", 8, "bold"), relief=tk.FLAT, command=self.generate_random_maze, cursor="hand2").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(m_row, text="🧹 Clear Walls", bg=DRACULA["dark_wall"], fg=DRACULA["fg"], font=("Segoe UI", 8, "bold"), relief=tk.FLAT, command=self.clear_maze_walls, cursor="hand2").pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        tk.Label(sidebar, text="Search Algorithm:", font=("Segoe UI", 8, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"]).pack(anchor="w")
        self.maze_algo_var = tk.StringVar(value="A* Search")
        ttk.Combobox(
            sidebar,
            textvariable=self.maze_algo_var,
            values=[
                "A* Search",
                "Bidirectional BFS",
                "UCS (Dijkstra)",
                "Breadth-First Search (BFS)",
                "Depth-First Search (DFS)",
                "Greedy Best-First Search",
            ],
            state="readonly",
        ).pack(fill=tk.X, pady=(2, 5))

        h_w_row = tk.Frame(sidebar, bg=DRACULA["current_line"])
        h_w_row.pack(fill=tk.X, pady=(0, 5))
        tk.Label(h_w_row, text="Heuristic:", font=("Segoe UI", 8, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"]).pack(side=tk.LEFT)
        self.maze_heur_var = tk.StringVar(value="Manhattan Distance")
        ttk.Combobox(h_w_row, textvariable=self.maze_heur_var, values=["Manhattan Distance", "Euclidean Distance"], state="readonly", width=18).pack(side=tk.RIGHT)

        tk.Label(sidebar, text="Weighted A* Factor (w = 1.0 Optimal .. 3.0 Fast):", font=("Segoe UI", 8, "bold"), bg=DRACULA["current_line"], fg=DRACULA["orange"]).pack(anchor="w")
        self.maze_weight_scale = tk.Scale(sidebar, from_=1.0, to=3.0, resolution=0.25, orient=tk.HORIZONTAL, bg=DRACULA["current_line"], fg=DRACULA["fg"], troughcolor=DRACULA["bg"], highlightthickness=0)
        self.maze_weight_scale.set(1.0)
        self.maze_weight_scale.pack(fill=tk.X, pady=(0, 4))

        self.maze_instant_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            sidebar,
            text="⚡ Instant Solution (Skip Animation)",
            variable=self.maze_instant_var,
            bg=DRACULA["current_line"],
            fg=DRACULA["yellow"],
            selectcolor=DRACULA["dark_wall"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", pady=(0, 2))

        self.maze_playback_mode_var = tk.StringVar(value="Explore + Path")
        mode_row = tk.Frame(sidebar, bg=DRACULA["current_line"])
        mode_row.pack(fill=tk.X, pady=(0, 5))
        tk.Label(mode_row, text="Replay View:", font=("Segoe UI", 8, "bold"), bg=DRACULA["current_line"], fg=DRACULA["fg"]).pack(side=tk.LEFT)
        ttk.Combobox(mode_row, textvariable=self.maze_playback_mode_var, values=["Explore + Path", "Path Only"], state="readonly", width=16).pack(side=tk.RIGHT)

        tk.Label(sidebar, text="⚡ Delay per Tile (ms) — Live Adjustable:", font=("Segoe UI", 8, "bold"), bg=DRACULA["current_line"], fg=DRACULA["yellow"]).pack(anchor="w")
        self.maze_speed_scale = tk.Scale(sidebar, from_=4, to=180, orient=tk.HORIZONTAL, bg=DRACULA["current_line"], fg=DRACULA["fg"], troughcolor=DRACULA["bg"], highlightthickness=0, activebackground=DRACULA["pink"])
        self.maze_speed_scale.set(28)
        self.maze_speed_scale.pack(fill=tk.X, pady=(0, 4))

        tk.Button(sidebar, text="🚀 Solve Maze", bg=DRACULA["green"], fg=DRACULA["bg"], font=("Segoe UI", 10, "bold"), relief=tk.FLAT, pady=4, command=self.solve_maze, cursor="hand2").pack(fill=tk.X, pady=2)

        p_row = tk.Frame(sidebar, bg=DRACULA["current_line"])
        p_row.pack(fill=tk.X, pady=2)
        tk.Button(p_row, text="🔁 Replay", bg=DRACULA["cyan"], fg=DRACULA["bg"], font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.replay_maze_animation, cursor="hand2").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.btn_maze_pause = tk.Button(p_row, text="⏸️ Pause", bg=DRACULA["orange"], fg=DRACULA["bg"], font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.toggle_maze_pause, cursor="hand2")
        self.btn_maze_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(p_row, text="⏮️ Reset", bg=DRACULA["red"], fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.reset_maze_animation, cursor="hand2").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(p_row, text="⏭️ Step", bg=DRACULA["purple"], fg=DRACULA["bg"], font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.step_maze_forward, cursor="hand2").pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        tk.Button(sidebar, text="📊 Benchmark All Solvers", bg=DRACULA["pink"], fg=DRACULA["bg"], font=("Segoe UI", 9, "bold"), relief=tk.FLAT, pady=3, command=self.compare_maze_algorithms).pack(fill=tk.X, pady=(6, 2))

        tk.Label(sidebar, text="💡 Left-Click: Wall | Right-Click: Erase", font=("Segoe UI", 8), bg=DRACULA["current_line"], fg=DRACULA["comment"]).pack(anchor="w")

        # --- Viewport ---
        kpi = tk.Frame(viewport, bg=DRACULA["current_line"], padx=12, pady=8)
        kpi.pack(fill=tk.X, pady=(0, 6))

        self.lbl_maze_status = tk.Label(kpi, text="Status: Ready", font=("Segoe UI", 10, "bold"), bg=DRACULA["current_line"], fg=DRACULA["green"])
        self.lbl_maze_status.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_maze_cost = tk.Label(kpi, text="Path: --", font=("Segoe UI", 10), bg=DRACULA["current_line"], fg=DRACULA["yellow"])
        self.lbl_maze_cost.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_maze_nodes = tk.Label(kpi, text="Step: 0 / 0", font=("Segoe UI", 10), bg=DRACULA["current_line"], fg=DRACULA["cyan"])
        self.lbl_maze_nodes.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_maze_time = tk.Label(kpi, text="Time: 0.0 ms", font=("Segoe UI", 10), bg=DRACULA["current_line"], fg=DRACULA["fg"])
        self.lbl_maze_time.pack(side=tk.LEFT)

        self.maze_canvas = tk.Canvas(viewport, bg=DRACULA["bg"], highlightthickness=0, cursor="crosshair")
        self.maze_canvas.pack(fill=tk.BOTH, expand=True)

        self.maze_canvas.bind("<Button-1>", self._on_maze_left_click)
        self.maze_canvas.bind("<B1-Motion>", self._on_maze_left_click)
        self.maze_canvas.bind("<Button-3>", self._on_maze_right_click)
        self.maze_canvas.bind("<B3-Motion>", self._on_maze_right_click)
        self.maze_canvas.bind("<Configure>", lambda e: self._redraw_maze())

    def _cancel_maze_timer(self):
        if self.maze_after_id is not None:
            self.root.after_cancel(self.maze_after_id)
            self.maze_after_id = None

    def _on_maze_left_click(self, event):
        if self.maze_animating:
            return
        cell = self._coords_to_maze_cell(event.x, event.y)
        if cell and cell != self.start and cell != self.goal:
            self.grid[cell[0], cell[1]] = 1
            self.maze_result = None
            self.maze_history.clear()
            self.maze_path.clear()
            self._redraw_maze()

    def _on_maze_right_click(self, event):
        if self.maze_animating:
            return
        cell = self._coords_to_maze_cell(event.x, event.y)
        if cell and cell != self.start and cell != self.goal:
            self.grid[cell[0], cell[1]] = 0
            self.maze_result = None
            self.maze_history.clear()
            self.maze_path.clear()
            self._redraw_maze()

    def _coords_to_maze_cell(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        w = self.maze_canvas.winfo_width()
        h = self.maze_canvas.winfo_height()
        grid_w = self.cols * self.cell_size
        grid_h = self.rows * self.cell_size
        offset_x = max(0, (w - grid_w) // 2)
        offset_y = max(0, (h - grid_h) // 2)
        c = (x - offset_x) // self.cell_size
        r = (y - offset_y) // self.cell_size
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return int(r), int(c)
        return None

    def _redraw_maze(self):
        self.maze_canvas.delete("all")
        w = self.maze_canvas.winfo_width()
        h = self.maze_canvas.winfo_height()
        if w <= 1 or h <= 1:
            return

        margin = 24
        self.cell_size = max(12, min(42, min((w - margin) // self.cols, (h - margin) // self.rows)))
        grid_w = self.cols * self.cell_size
        grid_h = self.rows * self.cell_size
        offset_x = max(0, (w - grid_w) // 2)
        offset_y = max(0, (h - grid_h) // 2)

        show_explore = self.maze_playback_mode_var.get() == "Explore + Path"
        visited_slice = self.maze_history[: self.maze_step] if show_explore else []
        visited_sides = {}
        for idx, cell in enumerate(visited_slice):
            side = self.maze_sides[idx] if (self.maze_sides and idx < len(self.maze_sides)) else 0
            visited_sides[cell] = side
        active_tile = visited_slice[-1] if (show_explore and visited_slice and self.maze_path_step == 0) else None

        visible_path = self.maze_path[: self.maze_path_step] if self.maze_path else []
        path_set = set(visible_path)
        path_head = visible_path[-1] if visible_path else None

        for r in range(self.rows):
            for c in range(self.cols):
                x1 = offset_x + c * self.cell_size
                y1 = offset_y + r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if (r, c) == self.start:
                    color = DRACULA["green"]
                elif (r, c) == self.goal:
                    color = DRACULA["red"]
                elif (r, c) == active_tile:
                    color = DRACULA["pink"]
                elif (r, c) in path_set:
                    color = DRACULA["yellow"]
                elif (r, c) in visited_sides:
                    color = DRACULA["bwd_blue"] if visited_sides[(r, c)] == 1 else DRACULA["purple"]
                elif self.grid[r, c] == 1:
                    color = DRACULA["dark_wall"]
                else:
                    color = DRACULA["current_line"]

                self.maze_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=DRACULA["comment"], width=1)

                lbl_font_sz = max(7, min(12, self.cell_size // 2))
                if (r, c) == self.start:
                    self.maze_canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2, text="S", fill=DRACULA["bg"], font=("Segoe UI", lbl_font_sz, "bold"))
                elif (r, c) == self.goal:
                    self.maze_canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2, text="G", fill=DRACULA["bg"], font=("Segoe UI", lbl_font_sz, "bold"))

        if len(visible_path) > 1:
            points = []
            for r, c in visible_path:
                points.extend([offset_x + c * self.cell_size + self.cell_size // 2, offset_y + r * self.cell_size + self.cell_size // 2])
            self.maze_canvas.create_line(points, fill=DRACULA["orange"], width=max(2, self.cell_size // 6), smooth=False)

        if path_head and self.maze_path_step < len(self.maze_path):
            hr, hc = path_head
            cx = offset_x + hc * self.cell_size + self.cell_size // 2
            cy = offset_y + hr * self.cell_size + self.cell_size // 2
            rad = max(3, self.cell_size // 3)
            self.maze_canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill=DRACULA["cyan"], outline="")

    def generate_random_maze(self):
        self._cancel_maze_timer()
        self.rows = int(self.maze_size_scale.get())
        self.cols = self.rows
        self.density = float(self.maze_density_scale.get()) / 100.0
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)

        gen = GridMaze.generate_solvable(
            self.rows,
            self.cols,
            density=self.density,
            preset=self.maze_preset_var.get(),
        )
        self.grid = gen.grid
        self.maze_result = None
        self.maze_history.clear()
        self.maze_sides = None
        self.maze_path.clear()
        self.reset_maze_animation()
        self.lbl_maze_status.config(text=f"Status: Solvable {self.rows}×{self.cols} Maze Ready", fg=DRACULA["cyan"])

    def clear_maze_walls(self):
        self._cancel_maze_timer()
        self.rows = int(self.maze_size_scale.get())
        self.cols = self.rows
        self.goal = (self.rows - 1, self.cols - 1)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.maze_result = None
        self.maze_history.clear()
        self.maze_sides = None
        self.maze_path.clear()
        self.reset_maze_animation()
        self.lbl_maze_status.config(text="Status: Walls Cleared", fg=DRACULA["cyan"])

    def _make_maze_agent(self, algo_name: str, metric_key: str, weight: float):
        if "Bidirectional BFS" in algo_name:
            return BidirectionalBFSAgent()
        if "UCS" in algo_name:
            return UCSAgent()
        if "A*" in algo_name:
            return AStarAgent(name=f"A* ({metric_key.title()})", heuristic_metric=metric_key, weight=weight)
        if "BFS" in algo_name:
            return BFSAgent()
        if "DFS" in algo_name:
            return DFSAgent()
        return GreedyAgent(name=f"Greedy ({metric_key.title()})", heuristic_metric=metric_key)

    def solve_maze(self):
        self._cancel_maze_timer()
        maze = GridMaze(grid=self.grid, start=self.start, goal=self.goal)
        metric_key = "manhattan" if "Manhattan" in self.maze_heur_var.get() else "euclidean"
        weight = float(self.maze_weight_scale.get())
        agent = self._make_maze_agent(self.maze_algo_var.get(), metric_key, weight)

        self.maze_result = agent.solve(maze)
        self.maze_history = list(self.maze_result.exploration_history)
        self.maze_sides = list(self.maze_result.exploration_sides) if self.maze_result.exploration_sides else None
        self.maze_path = list(self.maze_result.path) if self.maze_result.found else []
        self.lbl_maze_time.config(text=f"Time: {self.maze_result.execution_time_ms:.2f} ms")

        if self.maze_instant_var.get():
            self.maze_animating = False
            self.maze_paused = False
            self.maze_step = len(self.maze_history)
            self.maze_path_step = len(self.maze_path)
            self._redraw_maze()
            self._show_maze_final_status()
            return

        self.replay_maze_animation()

    def replay_maze_animation(self):
        if not self.maze_result:
            self.solve_maze()
            return
        self._cancel_maze_timer()
        show_explore = self.maze_playback_mode_var.get() == "Explore + Path"
        self.maze_step = 0 if show_explore else len(self.maze_history)
        self.maze_path_step = 0
        self.maze_animating = True
        self.maze_paused = False
        self.btn_maze_pause.config(text="⏸️ Pause")
        self.lbl_maze_status.config(text=f"Status: Replaying ({self.maze_result.algorithm_name})...", fg=DRACULA["yellow"])
        self._tick_maze_animation()

    def _show_maze_final_status(self):
        if self.maze_result and self.maze_result.found:
            self.lbl_maze_status.config(text="Status: ✅ Path Found!", fg=DRACULA["green"])
            self.lbl_maze_cost.config(text=f"Path: {self.maze_result.path_cost} steps")
            self.lbl_maze_nodes.config(text=f"Explored: {len(self.maze_history)}")
        else:
            self.lbl_maze_status.config(text="Status: ❌ Unsolvable Maze!", fg=DRACULA["red"])
            self.lbl_maze_cost.config(text="Path: N/A")

    def _tick_maze_animation(self):
        if not self.maze_animating or self.maze_paused:
            return
        show_explore = self.maze_playback_mode_var.get() == "Explore + Path"
        total_explore = len(self.maze_history)
        total_path = len(self.maze_path)
        delay = max(4, int(self.maze_speed_scale.get()))

        if show_explore and self.maze_step < total_explore:
            self.maze_step += 1
            self.lbl_maze_nodes.config(text=f"Exploring: {self.maze_step}/{total_explore}")
            self._redraw_maze()
            self.maze_after_id = self.root.after(delay, self._tick_maze_animation)
            return

        if self.maze_path_step < total_path:
            self.maze_path_step += 1
            self.lbl_maze_nodes.config(text=f"Path Step: {self.maze_path_step}/{total_path}")
            self._redraw_maze()
            path_delay = max(delay, min(140, int(delay * 1.35)))
            self.maze_after_id = self.root.after(path_delay, self._tick_maze_animation)
            return

        self.maze_animating = False
        self.maze_after_id = None
        self._redraw_maze()
        self._show_maze_final_status()

    def toggle_maze_pause(self):
        if not self.maze_animating:
            return
        self.maze_paused = not self.maze_paused
        if self.maze_paused:
            self._cancel_maze_timer()
            self.btn_maze_pause.config(text="▶️ Resume")
            self.lbl_maze_status.config(text="Status: Paused", fg=DRACULA["yellow"])
        else:
            self.btn_maze_pause.config(text="⏸️ Pause")
            self.lbl_maze_status.config(text="Status: Resumed...", fg=DRACULA["yellow"])
            self._tick_maze_animation()

    def reset_maze_animation(self):
        self._cancel_maze_timer()
        self.maze_animating = False
        self.maze_paused = False
        self.maze_step = 0
        self.maze_path_step = 0
        self.lbl_maze_status.config(text="Status: Reset", fg=DRACULA["cyan"])
        self.lbl_maze_cost.config(text="Path: --")
        self.lbl_maze_nodes.config(text="Step: 0 / 0")
        self.btn_maze_pause.config(text="⏸️ Pause")
        self._redraw_maze()

    def step_maze_forward(self):
        if not self.maze_history:
            return
        show_explore = self.maze_playback_mode_var.get() == "Explore + Path"
        total_explore = len(self.maze_history)
        total_path = len(self.maze_path)
        if show_explore and self.maze_step < total_explore:
            self.maze_step += 1
            self.lbl_maze_nodes.config(text=f"Exploring: {self.maze_step}/{total_explore}")
        elif self.maze_path_step < total_path:
            self.maze_path_step += 1
            self.lbl_maze_nodes.config(text=f"Path Step: {self.maze_path_step}/{total_path}")
        self._redraw_maze()
        if self.maze_path_step == total_path and (not show_explore or self.maze_step == total_explore):
            self._show_maze_final_status()

    def compare_maze_algorithms(self):
        maze = GridMaze(grid=self.grid, start=self.start, goal=self.goal)
        metric_key = "manhattan" if "Manhattan" in self.maze_heur_var.get() else "euclidean"
        agents = [
            AStarAgent(name=f"A* ({metric_key.title()})", heuristic_metric=metric_key),
            UCSAgent(name="UCS (Dijkstra)"),
            BFSAgent(name="BFS"),
            BidirectionalBFSAgent(name="Bidir BFS"),
            DFSAgent(name="DFS"),
            GreedyAgent(name=f"Greedy ({metric_key.title()})", heuristic_metric=metric_key),
        ]
        lines = [
            f"=== Benchmark Comparison ({self.rows}x{self.cols} Maze) ===",
            f"Heuristic: {metric_key.title()}\n",
            f"{'Algorithm':<20} | {'Found':<5} | {'Cost':<6} | {'Nodes':<6} | {'Time (ms)'}",
            "-" * 60,
        ]
        for ag in agents:
            res = ag.solve(maze)
            lines.append(f"{ag.name:<20} | {'Yes' if res.found else 'No':<5} | {str(res.path_cost) if res.found else 'N/A':<6} | {res.nodes_expanded:<6} | {res.execution_time_ms:.2f}")
        messagebox.showinfo("Algorithm Comparison Matrix", "\n".join(lines))

    # =====================================================
    # TAB 2: 8-Puzzle / 15-Puzzle Implementation
    # =====================================================
    def _build_puzzle_tab(self):
        container = tk.Frame(self.tab_puzzle, bg=DRACULA["bg"])
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        sidebar = tk.Frame(container, bg=DRACULA["current_line"], width=330, padx=12, pady=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar.pack_propagate(False)

        viewport = tk.Frame(container, bg=DRACULA["bg"])
        viewport.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(sidebar, text="Board Size:", font=("Segoe UI", 8, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"]).pack(anchor="w")
        self.puzzle_size_var = tk.StringVar(value="3×3 (8-Puzzle)")
        ttk.Combobox(sidebar, textvariable=self.puzzle_size_var, values=["3×3 (8-Puzzle)", "4×4 (15-Puzzle)"], state="readonly").pack(fill=tk.X, pady=(2, 5))

        tk.Label(sidebar, text="Preset Configuration:", font=("Segoe UI", 8, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"]).pack(anchor="w")
        self.puzzle_preset_var = tk.StringVar(value="Random Solvable Board")
        ttk.Combobox(sidebar, textvariable=self.puzzle_preset_var, values=["Random Solvable Board", "1-Move Trivial Solution", "3-Moves Easy", "6-Moves Medium", "10-Moves Challenging"], state="readonly").pack(fill=tk.X, pady=(2, 5))

        tk.Button(sidebar, text="🎲 Scramble / Apply Board", bg=DRACULA["comment"], fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.apply_puzzle_preset).pack(fill=tk.X, pady=(0, 6))

        self.lbl_p_solv = tk.Label(sidebar, text="Solvability: Checking...", font=("Segoe UI", 8), justify=tk.LEFT, bg=DRACULA["dark_wall"], fg=DRACULA["green"], padx=8, pady=5)
        self.lbl_p_solv.pack(fill=tk.X, pady=(0, 6))

        tk.Label(sidebar, text="Heuristic Metric:", font=("Segoe UI", 8, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"]).pack(anchor="w")
        self.puzzle_heur_var = tk.StringVar(value="Manhattan")
        ttk.Combobox(sidebar, textvariable=self.puzzle_heur_var, values=["Manhattan", "Euclidean"], state="readonly").pack(fill=tk.X, pady=(2, 5))

        tk.Label(sidebar, text="Search Strategy:", font=("Segoe UI", 8, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"]).pack(anchor="w")
        self.puzzle_algo_var = tk.StringVar(value="A* Search")
        ttk.Combobox(sidebar, textvariable=self.puzzle_algo_var, values=["A* Search", "Bidirectional BFS", "UCS (Dijkstra)", "Greedy Best-First", "Breadth-First Search (BFS)", "Depth-First Search (DFS)"], state="readonly").pack(fill=tk.X, pady=(2, 5))

        self.puzzle_instant_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sidebar, text="⚡ Instant Solution (Skip Animation)", variable=self.puzzle_instant_var, bg=DRACULA["current_line"], fg=DRACULA["yellow"], selectcolor=DRACULA["dark_wall"], font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 4))

        tk.Label(sidebar, text="⚡ Move Delay (ms):", font=("Segoe UI", 8, "bold"), bg=DRACULA["current_line"], fg=DRACULA["yellow"]).pack(anchor="w")
        self.puzzle_speed_scale = tk.Scale(sidebar, from_=40, to=600, orient=tk.HORIZONTAL, bg=DRACULA["current_line"], fg=DRACULA["fg"], troughcolor=DRACULA["bg"], highlightthickness=0, activebackground=DRACULA["pink"])
        self.puzzle_speed_scale.set(220)
        self.puzzle_speed_scale.pack(fill=tk.X, pady=(0, 4))

        tk.Button(sidebar, text="🚀 Solve Puzzle", bg=DRACULA["green"], fg=DRACULA["bg"], font=("Segoe UI", 10, "bold"), relief=tk.FLAT, pady=4, command=self.solve_puzzle, cursor="hand2").pack(fill=tk.X, pady=2)

        p_btn_row = tk.Frame(sidebar, bg=DRACULA["current_line"])
        p_btn_row.pack(fill=tk.X, pady=3)
        tk.Button(p_btn_row, text="🔁 Replay", bg=DRACULA["cyan"], fg=DRACULA["bg"], font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.replay_puzzle_animation, cursor="hand2").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.btn_puzzle_pause = tk.Button(p_btn_row, text="⏸️ Pause", bg=DRACULA["orange"], fg=DRACULA["bg"], font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.toggle_puzzle_pause, cursor="hand2")
        self.btn_puzzle_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(p_btn_row, text="⏮️ Reset", bg=DRACULA["red"], fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.reset_puzzle_animation, cursor="hand2").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(p_btn_row, text="⏭️ Step", bg=DRACULA["purple"], fg=DRACULA["bg"], font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.step_puzzle_forward, cursor="hand2").pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        self.lbl_p_move_desc = tk.Label(sidebar, text="Action: Initial Board State", font=("Segoe UI", 9, "bold"), bg=DRACULA["dark_wall"], fg=DRACULA["pink"], padx=8, pady=6, wraplength=290, justify=tk.LEFT)
        self.lbl_p_move_desc.pack(fill=tk.X, pady=(8, 0))

        p_kpi = tk.Frame(viewport, bg=DRACULA["current_line"], padx=12, pady=8)
        p_kpi.pack(fill=tk.X, pady=(0, 6))

        self.lbl_p_status = tk.Label(p_kpi, text="Status: Ready", font=("Segoe UI", 10, "bold"), bg=DRACULA["current_line"], fg=DRACULA["green"])
        self.lbl_p_status.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_p_moves = tk.Label(p_kpi, text="Moves: --", font=("Segoe UI", 10), bg=DRACULA["current_line"], fg=DRACULA["yellow"])
        self.lbl_p_moves.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_p_nodes = tk.Label(p_kpi, text="Nodes Expanded: 0", font=("Segoe UI", 10), bg=DRACULA["current_line"], fg=DRACULA["cyan"])
        self.lbl_p_nodes.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_p_time = tk.Label(p_kpi, text="Time: 0.0 ms", font=("Segoe UI", 10), bg=DRACULA["current_line"], fg=DRACULA["fg"])
        self.lbl_p_time.pack(side=tk.LEFT)

        self.puzzle_canvas = tk.Canvas(viewport, bg=DRACULA["bg"], highlightthickness=0)
        self.puzzle_canvas.pack(fill=tk.BOTH, expand=True)
        self.puzzle_canvas.bind("<Configure>", lambda e: self._redraw_puzzle())

        self._update_puzzle_solvability_label()
        self._redraw_puzzle()

    def _cancel_puzzle_timer(self):
        if self.puzzle_after_id is not None:
            self.root.after_cancel(self.puzzle_after_id)
            self.puzzle_after_id = None

    def apply_puzzle_preset(self):
        is_15 = "4×4" in self.puzzle_size_var.get()
        self.puzzle_dim = 4 if is_15 else 3
        self.puzzle_goal = EightPuzzle.STANDARD_GOAL_15 if is_15 else EightPuzzle.STANDARD_GOAL
        pre = self.puzzle_preset_var.get()

        if not is_15 and pre in ("1-Move Trivial Solution", "3-Moves Easy", "6-Moves Medium", "10-Moves Challenging"):
            fixed = {
                "1-Move Trivial Solution": (1, 2, 3, 4, 5, 6, 7, 0, 8),
                "3-Moves Easy": (1, 2, 3, 4, 0, 6, 7, 5, 8),
                "6-Moves Medium": (1, 2, 3, 0, 4, 6, 7, 5, 8),
                "10-Moves Challenging": (1, 3, 6, 4, 2, 5, 7, 0, 8),
            }
            self.puzzle_state = fixed[pre]
        else:
            depth = {"1-Move Trivial Solution": 1, "3-Moves Easy": 3, "6-Moves Medium": 6, "10-Moves Challenging": 10}.get(pre, 16 if is_15 else 18)
            self.puzzle_state = EightPuzzle.generate_random_solvable(goal_state=self.puzzle_goal, scramble_moves=depth)

        self.puzzle_result = None
        self.puzzle_path.clear()
        self.puzzle_moves_info.clear()
        self.reset_puzzle_animation()
        self._update_puzzle_solvability_label()
        self._redraw_puzzle()

    def _update_puzzle_solvability_label(self):
        inv = EightPuzzle.count_inversions(self.puzzle_state)
        solv = EightPuzzle.is_solvable(self.puzzle_state, self.puzzle_goal)
        status_txt = "🟢 Solvable" if solv else "🔴 Unsolvable"
        self.lbl_p_solv.config(
            text=f"Parity Check ({self.puzzle_dim}×{self.puzzle_dim}):\n• Inversions: {inv} | BlankRow: {self.puzzle_state.index(0) // self.puzzle_dim}\n• Status: {status_txt}",
            fg=DRACULA["green"] if solv else DRACULA["red"],
        )

    def _redraw_puzzle(self):
        self.puzzle_canvas.delete("all")
        w = self.puzzle_canvas.winfo_width()
        h = self.puzzle_canvas.winfo_height()
        if w <= 1 or h <= 1:
            return

        dim = self.puzzle_dim
        board_size = min(w - 60, h - 60, 460)
        tile_size = board_size // dim
        offset_x = (w - (tile_size * dim)) // 2
        offset_y = (h - (tile_size * dim)) // 2

        curr_state = (
            self.puzzle_path[self.puzzle_step]
            if self.puzzle_path and self.puzzle_step < len(self.puzzle_path)
            else self.puzzle_state
        )
        moved_tile = (
            self.puzzle_moves_info[self.puzzle_step][0]
            if self.puzzle_moves_info and self.puzzle_step < len(self.puzzle_moves_info)
            else None
        )

        font_sz = 26 if dim == 3 else 20
        for idx, val in enumerate(curr_state):
            r, c = divmod(idx, dim)
            x1, y1 = offset_x + c * tile_size, offset_y + r * tile_size
            x2, y2 = x1 + tile_size, y1 + tile_size
            if val == 0:
                self.puzzle_canvas.create_rectangle(x1 + 4, y1 + 4, x2 - 4, y2 - 4, fill=DRACULA["dark_wall"], outline=DRACULA["comment"], width=2, dash=(4, 4))
            else:
                is_moved = moved_tile is not None and val == moved_tile
                self.puzzle_canvas.create_rectangle(
                    x1 + 4, y1 + 4, x2 - 4, y2 - 4,
                    fill=DRACULA["pink"] if is_moved else DRACULA["current_line"],
                    outline=DRACULA["yellow"] if is_moved else DRACULA["purple"],
                    width=3 if is_moved else 2,
                )
                self.puzzle_canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2, text=str(val), fill=DRACULA["bg"] if is_moved else DRACULA["cyan"], font=("Segoe UI", font_sz, "bold"))

    def solve_puzzle(self):
        self._cancel_puzzle_timer()
        env = EightPuzzle(start_state=self.puzzle_state, goal_state=self.puzzle_goal)
        h_choice = self.puzzle_heur_var.get()
        metric_key = "euclidean" if "Euclidean" in h_choice else "manhattan"
        algo = self.puzzle_algo_var.get()

        if "Bidirectional BFS" in algo:
            agent = BidirectionalBFSAgent()
        elif "UCS" in algo:
            agent = UCSAgent()
        elif "A*" in algo:
            agent = AStarAgent(name=f"A* ({h_choice})", heuristic_metric=metric_key)
        elif "Greedy" in algo:
            agent = GreedyAgent(name=f"Greedy ({h_choice})", heuristic_metric=metric_key)
        elif "BFS" in algo:
            agent = BFSAgent()
        else:
            agent = DFSAgent()

        self.puzzle_result = agent.solve(env)
        if not self.puzzle_result.found:
            messagebox.showerror("No Solution", "This puzzle configuration has no path to goal.")
            return

        dim = self.puzzle_dim
        self.puzzle_path = list(self.puzzle_result.path)
        self.puzzle_moves_info = [(None, "Initial Board State")]
        for idx in range(1, len(self.puzzle_path)):
            prev_b, curr_b = self.puzzle_path[idx - 1], self.puzzle_path[idx]
            pz, cz = prev_b.index(0), curr_b.index(0)
            moved_t = prev_b[cz]
            r0, c0 = divmod(cz, dim)
            r1, c1 = divmod(pz, dim)
            dr, dc = r1 - r0, c1 - c0
            direction = "UP" if dr == -1 else ("DOWN" if dr == 1 else ("LEFT" if dc == -1 else "RIGHT"))
            self.puzzle_moves_info.append((moved_t, f"Step {idx}: Tile {moved_t} slid {direction} into blank slot"))

        self.lbl_p_moves.config(text=f"Moves: {self.puzzle_result.path_cost}")
        self.lbl_p_nodes.config(text=f"Nodes Expanded: {self.puzzle_result.nodes_expanded}")
        self.lbl_p_time.config(text=f"Time: {self.puzzle_result.execution_time_ms:.2f} ms")

        if self.puzzle_instant_var.get():
            self.puzzle_animating = False
            self.puzzle_paused = False
            self.puzzle_step = len(self.puzzle_path) - 1
            self._redraw_puzzle()
            self.lbl_p_status.config(text="Status: ✅ Goal Board Reached!", fg=DRACULA["green"])
            self.lbl_p_move_desc.config(text="🎉 Solution Complete! Goal state matched.")
            return

        self.replay_puzzle_animation()

    def replay_puzzle_animation(self):
        if not self.puzzle_path:
            self.solve_puzzle()
            return
        self._cancel_puzzle_timer()
        self.puzzle_step = 0
        self.puzzle_animating = True
        self.puzzle_paused = False
        self.btn_puzzle_pause.config(text="⏸️ Pause")
        self.lbl_p_status.config(text="Status: Replaying...", fg=DRACULA["yellow"])
        self._tick_puzzle_animation()

    def _tick_puzzle_animation(self):
        if not self.puzzle_animating or self.puzzle_paused:
            return
        total = len(self.puzzle_path)
        if self.puzzle_step < total:
            delay = max(40, int(self.puzzle_speed_scale.get()))
            _, desc = self.puzzle_moves_info[self.puzzle_step]
            self.lbl_p_move_desc.config(text=f"📍 {desc}")
            self.lbl_p_status.config(text=f"Status: Move {self.puzzle_step}/{total - 1}", fg=DRACULA["cyan"])
            self._redraw_puzzle()
            self.puzzle_step += 1
            self.puzzle_after_id = self.root.after(delay, self._tick_puzzle_animation)
        else:
            self.puzzle_animating = False
            self.puzzle_after_id = None
            self.puzzle_step = total - 1
            self._redraw_puzzle()
            self.lbl_p_status.config(text="Status: ✅ Goal Board Reached!", fg=DRACULA["green"])
            self.lbl_p_move_desc.config(text="🎉 Solution Complete! Goal state matched.")

    def toggle_puzzle_pause(self):
        if not self.puzzle_animating:
            return
        self.puzzle_paused = not self.puzzle_paused
        if self.puzzle_paused:
            self._cancel_puzzle_timer()
            self.btn_puzzle_pause.config(text="▶️ Resume")
            self.lbl_p_status.config(text="Status: Paused", fg=DRACULA["yellow"])
        else:
            self.btn_puzzle_pause.config(text="⏸️ Pause")
            self.lbl_p_status.config(text="Status: Resumed...", fg=DRACULA["yellow"])
            self._tick_puzzle_animation()

    def reset_puzzle_animation(self):
        self._cancel_puzzle_timer()
        self.puzzle_animating = False
        self.puzzle_paused = False
        self.puzzle_step = 0
        self.lbl_p_status.config(text="Status: Ready", fg=DRACULA["green"])
        self.lbl_p_moves.config(text="Moves: --")
        self.lbl_p_nodes.config(text="Nodes Expanded: 0")
        self.lbl_p_move_desc.config(text="Action: Initial Board State")
        self.btn_puzzle_pause.config(text="⏸️ Pause")
        self._redraw_puzzle()

    def step_puzzle_forward(self):
        if not self.puzzle_path:
            return
        total = len(self.puzzle_path)
        if self.puzzle_step < total - 1:
            self.puzzle_step += 1
            _, desc = self.puzzle_moves_info[self.puzzle_step]
            self.lbl_p_move_desc.config(text=f"📍 {desc}")
            self.lbl_p_status.config(text=f"Status: Step {self.puzzle_step}/{total - 1}", fg=DRACULA["cyan"])
            self._redraw_puzzle()
            if self.puzzle_step == total - 1:
                self.lbl_p_status.config(text="Status: ✅ Goal Reached!", fg=DRACULA["green"])


def launch_gui():
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print("\n[MazeMind AI Desktop GUI Error]")
        print("Could not initialize graphical display (Tkinter TclError):", e)
        print("Alternatively, launch the web GUI: uv run streamlit run app.py\n")
        sys.exit(1)

    MazeDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
