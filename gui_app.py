"""
MazeMind AI — Native Desktop GUI Application (Dracula Theme).
Specifications: https://draculatheme.com/spec

Features:
- Dual Problem Modes: 2D Grid Maze & 8-Puzzle Sliding Tiles
- 1-by-1 Animated Tile Movements for 8-Puzzle with Move Descriptions & Highlights
- Complete Playback Controls: Play, Pause, Reset, Step, Speed Up / Slow Down
- Real-time 60 FPS Canvas Animations with Speed Slider (1ms to 500ms)
- Interactive Maze Editing (Left-click to place walls, Right-click to erase)
- Both Manhattan and Euclidean Distance Heuristics
- Full OOP integration with `src/mazemind` (BFS, DFS, Greedy, A*)
Course: CSL-325 Artificial Intelligence Lab (Fall 2026).
"""

import math
import os
import random
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from mazemind.environments import GridMaze, EightPuzzle
from mazemind.search_agents import (
    SearchResult,
    BFSAgent,
    DFSAgent,
    GreedyAgent,
    AStarAgent,
)

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


class MazeDesktopApp:
    """
    Native Desktop GUI for MazeMind AI with Dracula Theme,
    interactive drawing, and real-time step-by-step animations.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MazeMind AI — Intelligent Puzzle Solver & Search Visualizer (Dracula Theme)")
        self.root.geometry("1150x780")
        self.root.minsize(1000, 700)
        self.root.configure(bg=DRACULA["bg"])

        # ---------------- Maze State ----------------
        self.rows = 15
        self.cols = 15
        self.cell_size = 32
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int32)

        self.maze_animating = False
        self.maze_paused = False
        self.maze_step = 0
        self.maze_history: List[Tuple[int, int]] = []
        self.maze_path: List[Tuple[int, int]] = []
        self.maze_result: Optional[SearchResult] = None

        # ---------------- 8-Puzzle State ----------------
        self.puzzle_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)
        self.puzzle_goal = EightPuzzle.STANDARD_GOAL
        self.puzzle_animating = False
        self.puzzle_paused = False
        self.puzzle_step = 0
        self.puzzle_path: List[Tuple[int, ...]] = []
        self.puzzle_moves_info: List[Tuple[Optional[int], str]] = []
        self.puzzle_result: Optional[SearchResult] = None

        # Setup Styles & UI
        self._setup_styles()
        self._build_ui()
        self._redraw_maze()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Dracula ttk styling
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
        # Top Header Bar
        header = tk.Frame(self.root, bg=DRACULA["current_line"], padx=16, pady=10)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="🧠 MazeMind AI",
            font=("Segoe UI", 16, "bold"),
            bg=DRACULA["current_line"],
            fg=DRACULA["purple"],
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = tk.Label(
            header,
            text="Intelligent Puzzle Solver & Search Visualizer (CSL-325 AI Lab • Dracula Theme)",
            font=("Segoe UI", 10),
            bg=DRACULA["current_line"],
            fg=DRACULA["cyan"],
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=(12, 0))

        team_lbl = tk.Label(
            header,
            text="BS (CS) 5-B • Samiullah, Musavir, Farman",
            font=("Segoe UI", 9, "bold"),
            bg=DRACULA["current_line"],
            fg=DRACULA["green"],
        )
        team_lbl.pack(side=tk.RIGHT)

        # Tab Notebook (2D Grid Maze & 8-Puzzle)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Tab 1: 2D Grid Maze
        self.tab_maze = tk.Frame(self.notebook, bg=DRACULA["bg"])
        self.notebook.add(self.tab_maze, text=" 🗺️ 2D Grid Maze Solver ")

        # Tab 2: 8-Puzzle
        self.tab_puzzle = tk.Frame(self.notebook, bg=DRACULA["bg"])
        self.notebook.add(self.tab_puzzle, text=" 🧩 8-Puzzle Sliding Tiles ")

        self._build_maze_tab()
        self._build_puzzle_tab()

    # =====================================================
    # TAB 1: 2D Grid Maze Implementation
    # =====================================================
    def _build_maze_tab(self):
        # Container
        container = tk.Frame(self.tab_maze, bg=DRACULA["bg"])
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left Sidebar (Controls)
        sidebar = tk.Frame(container, bg=DRACULA["current_line"], width=310, padx=12, pady=12)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar.pack_propagate(False)

        # Right Viewport (Canvas + Status)
        viewport = tk.Frame(container, bg=DRACULA["bg"])
        viewport.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Sidebar Controls ---
        lbl_h = tk.Label(sidebar, text="Heuristic Distance Function:", font=("Segoe UI", 9, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"])
        lbl_h.pack(anchor="w", pady=(2, 2))

        self.maze_heur_var = tk.StringVar(value="Manhattan Distance")
        combo_h = ttk.Combobox(sidebar, textvariable=self.maze_heur_var, values=["Manhattan Distance", "Euclidean Distance"], state="readonly")
        combo_h.pack(fill=tk.X, pady=(0, 8))

        lbl_a = tk.Label(sidebar, text="Search Algorithm:", font=("Segoe UI", 9, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"])
        lbl_a.pack(anchor="w", pady=(2, 2))

        self.maze_algo_var = tk.StringVar(value="A* Search")
        combo_a = ttk.Combobox(sidebar, textvariable=self.maze_algo_var, values=["A* Search", "Breadth-First Search (BFS)", "Depth-First Search (DFS)", "Greedy Best-First Search"], state="readonly")
        combo_a.pack(fill=tk.X, pady=(0, 10))

        # Animation Speed Slider & Buttons
        lbl_sp = tk.Label(sidebar, text="⚡ Animation Delay (ms):", font=("Segoe UI", 9, "bold"), bg=DRACULA["current_line"], fg=DRACULA["yellow"])
        lbl_sp.pack(anchor="w", pady=(2, 2))

        self.maze_speed_scale = tk.Scale(sidebar, from_=1, to=150, orient=tk.HORIZONTAL, bg=DRACULA["current_line"], fg=DRACULA["fg"], troughcolor=DRACULA["bg"], highlightthickness=0, activebackground=DRACULA["pink"])
        self.maze_speed_scale.set(15)
        self.maze_speed_scale.pack(fill=tk.X, pady=(0, 4))

        speed_btn_row = tk.Frame(sidebar, bg=DRACULA["current_line"])
        speed_btn_row.pack(fill=tk.X, pady=(0, 10))

        btn_slower = tk.Button(speed_btn_row, text="🐢 Slower (+20ms)", bg=DRACULA["comment"], fg="white", font=("Segoe UI", 8), relief=tk.FLAT, command=lambda: self.adjust_maze_speed(20))
        btn_slower.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        btn_faster = tk.Button(speed_btn_row, text="🚀 Faster (-10ms)", bg=DRACULA["comment"], fg="white", font=("Segoe UI", 8), relief=tk.FLAT, command=lambda: self.adjust_maze_speed(-10))
        btn_faster.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Playback Buttons: Play / Pause / Reset / Step
        lbl_ctrl = tk.Label(sidebar, text="⏯️ Playback Controls:", font=("Segoe UI", 9, "bold"), bg=DRACULA["current_line"], fg=DRACULA["pink"])
        lbl_ctrl.pack(anchor="w", pady=(4, 2))

        self.btn_maze_play = tk.Button(sidebar, text="▶️ Play / Solve", bg=DRACULA["green"], fg=DRACULA["bg"], font=("Segoe UI", 10, "bold"), relief=tk.FLAT, pady=5, command=self.play_maze_animation, cursor="hand2")
        self.btn_maze_play.pack(fill=tk.X, pady=2)

        p_row = tk.Frame(sidebar, bg=DRACULA["current_line"])
        p_row.pack(fill=tk.X, pady=2)

        self.btn_maze_pause = tk.Button(p_row, text="⏸️ Pause", bg=DRACULA["orange"], fg=DRACULA["bg"], font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.toggle_maze_pause, cursor="hand2")
        self.btn_maze_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.btn_maze_reset = tk.Button(p_row, text="⏮️ Reset", bg=DRACULA["red"], fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.reset_maze_animation, cursor="hand2")
        self.btn_maze_reset.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_maze_step = tk.Button(p_row, text="⏭️ Step", bg=DRACULA["purple"], fg=DRACULA["bg"], font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.step_maze_forward, cursor="hand2")
        self.btn_maze_step.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Maze Layout Buttons
        lbl_m = tk.Label(sidebar, text="Grid Actions:", font=("Segoe UI", 9, "bold"), bg=DRACULA["current_line"], fg=DRACULA["fg"])
        lbl_m.pack(anchor="w", pady=(10, 2))

        m_row = tk.Frame(sidebar, bg=DRACULA["current_line"])
        m_row.pack(fill=tk.X, pady=2)

        btn_rand = tk.Button(m_row, text="🎲 Random Maze", bg=DRACULA["comment"], fg="white", font=("Segoe UI", 8, "bold"), relief=tk.FLAT, command=self.generate_random_maze)
        btn_rand.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        btn_clr = tk.Button(m_row, text="🧹 Clear Walls", bg=DRACULA["dark_wall"], fg=DRACULA["fg"], font=("Segoe UI", 8, "bold"), relief=tk.FLAT, command=self.clear_maze_walls)
        btn_clr.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        btn_comp = tk.Button(sidebar, text="📊 Benchmark All 4 Algorithms", bg=DRACULA["pink"], fg=DRACULA["bg"], font=("Segoe UI", 9, "bold"), relief=tk.FLAT, pady=4, command=self.compare_maze_algorithms)
        btn_comp.pack(fill=tk.X, pady=(10, 4))

        # Instructions
        lbl_inst = tk.Label(
            sidebar,
            text="💡 Mouse Guide:\n• Left-Click/Drag: Draw Wall\n• Right-Click/Drag: Erase Wall",
            font=("Segoe UI", 8),
            justify=tk.LEFT,
            bg=DRACULA["current_line"],
            fg=DRACULA["comment"],
        )
        lbl_inst.pack(anchor="w", pady=(10, 0))

        # --- Viewport ---
        kpi = tk.Frame(viewport, bg=DRACULA["current_line"], padx=12, pady=8)
        kpi.pack(fill=tk.X, pady=(0, 6))

        self.lbl_maze_status = tk.Label(kpi, text="Status: Ready", font=("Segoe UI", 10, "bold"), bg=DRACULA["current_line"], fg=DRACULA["green"])
        self.lbl_maze_status.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_maze_cost = tk.Label(kpi, text="Path Length: --", font=("Segoe UI", 10), bg=DRACULA["current_line"], fg=DRACULA["yellow"])
        self.lbl_maze_cost.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_maze_nodes = tk.Label(kpi, text="Nodes Visited: 0", font=("Segoe UI", 10), bg=DRACULA["current_line"], fg=DRACULA["cyan"])
        self.lbl_maze_nodes.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_maze_time = tk.Label(kpi, text="Time: 0.0 ms", font=("Segoe UI", 10), bg=DRACULA["current_line"], fg=DRACULA["fg"])
        self.lbl_maze_time.pack(side=tk.LEFT)

        # Canvas
        self.maze_canvas = tk.Canvas(viewport, bg=DRACULA["bg"], highlightthickness=0, cursor="crosshair")
        self.maze_canvas.pack(fill=tk.BOTH, expand=True)

        self.maze_canvas.bind("<Button-1>", self._on_maze_left_click)
        self.maze_canvas.bind("<B1-Motion>", self._on_maze_left_click)
        self.maze_canvas.bind("<Button-3>", self._on_maze_right_click)
        self.maze_canvas.bind("<B3-Motion>", self._on_maze_right_click)
        self.maze_canvas.bind("<Configure>", lambda e: self._redraw_maze())

    def adjust_maze_speed(self, delta: int):
        cur = self.maze_speed_scale.get()
        self.maze_speed_scale.set(max(1, min(150, cur + delta)))

    def _on_maze_left_click(self, event):
        if self.maze_animating:
            return
        cell = self._coords_to_maze_cell(event.x, event.y)
        if cell and cell != self.start and cell != self.goal:
            self.grid[cell[0], cell[1]] = 1
            self._redraw_maze()

    def _on_maze_right_click(self, event):
        if self.maze_animating:
            return
        cell = self._coords_to_maze_cell(event.x, event.y)
        if cell and cell != self.start and cell != self.goal:
            self.grid[cell[0], cell[1]] = 0
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

        margin = 30
        max_cw = (w - margin) // self.cols
        max_ch = (h - margin) // self.rows
        self.cell_size = max(18, min(45, min(max_cw, max_ch)))

        grid_w = self.cols * self.cell_size
        grid_h = self.rows * self.cell_size
        offset_x = max(0, (w - grid_w) // 2)
        offset_y = max(0, (h - grid_h) // 2)

        visited_set = set(self.maze_history[: self.maze_step])
        path_set = set(self.maze_path) if self.maze_step >= len(self.maze_history) and self.maze_path else set()

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
                elif (r, c) in path_set:
                    color = DRACULA["yellow"]
                elif (r, c) in visited_set:
                    color = DRACULA["purple"]
                elif self.grid[r, c] == 1:
                    color = DRACULA["dark_wall"]
                else:
                    color = DRACULA["current_line"]

                self.maze_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=DRACULA["comment"], width=1)

                if (r, c) == self.start:
                    self.maze_canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2, text="S", fill=DRACULA["bg"], font=("Segoe UI", 12, "bold"))
                elif (r, c) == self.goal:
                    self.maze_canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2, text="G", fill=DRACULA["bg"], font=("Segoe UI", 12, "bold"))

        if path_set and len(self.maze_path) > 1:
            points = []
            for r, c in self.maze_path:
                px = offset_x + c * self.cell_size + self.cell_size // 2
                py = offset_y + r * self.cell_size + self.cell_size // 2
                points.extend([px, py])
            self.maze_canvas.create_line(points, fill=DRACULA["orange"], width=3, smooth=True)

    def generate_random_maze(self):
        if self.maze_animating:
            return
        rng = np.random.default_rng()
        self.grid = (rng.random((self.rows, self.cols)) < 0.24).astype(np.int32)
        self.grid[self.start[0], self.start[1]] = 0
        self.grid[self.goal[0], self.goal[1]] = 0
        self.reset_maze_animation()
        self.lbl_maze_status.config(text="Status: Random Maze Created", fg=DRACULA["cyan"])

    def clear_maze_walls(self):
        if self.maze_animating:
            return
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.reset_maze_animation()
        self.lbl_maze_status.config(text="Status: Walls Cleared", fg=DRACULA["cyan"])

    def play_maze_animation(self):
        if self.maze_animating and not self.maze_paused:
            return

        if self.maze_paused:
            self.maze_paused = False
            self.btn_maze_pause.config(text="⏸️ Pause")
            self.lbl_maze_status.config(text="Status: Resumed...", fg=DRACULA["yellow"])
            self._tick_maze_animation()
            return

        maze = GridMaze(grid=self.grid, start=self.start, goal=self.goal)
        metric_key = "manhattan" if "Manhattan" in self.maze_heur_var.get() else "euclidean"
        algo = self.maze_algo_var.get()

        if "A*" in algo:
            agent = AStarAgent(name=f"A* ({metric_key.title()})", heuristic_metric=metric_key)
        elif "BFS" in algo:
            agent = BFSAgent(name="BFS")
        elif "DFS" in algo:
            agent = DFSAgent(name="DFS")
        else:
            agent = GreedyAgent(name=f"Greedy ({metric_key.title()})", heuristic_metric=metric_key)

        self.maze_result = agent.solve(maze)
        self.maze_history = self.maze_result.exploration_history
        self.maze_path = self.maze_result.path if self.maze_result.found else []
        self.maze_step = 0

        self.maze_animating = True
        self.maze_paused = False
        self.btn_maze_pause.config(text="⏸️ Pause")
        self.lbl_maze_status.config(text=f"Status: Searching ({agent.name})...", fg=DRACULA["yellow"])
        self.lbl_maze_time.config(text=f"Time: {self.maze_result.execution_time_ms:.2f} ms")

        self._tick_maze_animation()

    def _tick_maze_animation(self):
        if not self.maze_animating or self.maze_paused:
            return

        total = len(self.maze_history)
        if self.maze_step < total:
            delay = max(1, int(self.maze_speed_scale.get()))
            stride = 1 if delay > 30 else (2 if delay > 10 else 4)

            self.maze_step = min(total, self.maze_step + stride)
            self.lbl_maze_nodes.config(text=f"Nodes Visited: {self.maze_step}/{total}")
            self._redraw_maze()
            self.root.after(delay, self._tick_maze_animation)
        else:
            self.maze_animating = False
            self._redraw_maze()
            if self.maze_result and self.maze_result.found:
                self.lbl_maze_status.config(text="Status: ✅ Path Found!", fg=DRACULA["green"])
                self.lbl_maze_cost.config(text=f"Path Length: {self.maze_result.path_cost} steps")
            else:
                self.lbl_maze_status.config(text="Status: ❌ Unsolvable Maze!", fg=DRACULA["red"])
                self.lbl_maze_cost.config(text="Path Length: N/A")

    def toggle_maze_pause(self):
        if not self.maze_animating:
            return
        self.maze_paused = not self.maze_paused
        if self.maze_paused:
            self.btn_maze_pause.config(text="▶️ Resume")
            self.lbl_maze_status.config(text="Status: Paused", fg=DRACULA["yellow"])
        else:
            self.btn_maze_pause.config(text="⏸️ Pause")
            self.lbl_maze_status.config(text="Status: Resumed...", fg=DRACULA["yellow"])
            self._tick_maze_animation()

    def reset_maze_animation(self):
        self.maze_animating = False
        self.maze_paused = False
        self.maze_step = 0
        self.maze_history.clear()
        self.maze_path.clear()
        self.lbl_maze_status.config(text="Status: Reset", fg=DRACULA["cyan"])
        self.lbl_maze_cost.config(text="Path Length: --")
        self.lbl_maze_nodes.config(text="Nodes Visited: 0")
        self.lbl_maze_time.config(text="Time: 0.0 ms")
        self.btn_maze_pause.config(text="⏸️ Pause")
        self._redraw_maze()

    def step_maze_forward(self):
        if not self.maze_history:
            return
        total = len(self.maze_history)
        if self.maze_step < total:
            self.maze_step = min(total, self.maze_step + 1)
            self.lbl_maze_nodes.config(text=f"Nodes Visited: {self.maze_step}/{total}")
            self._redraw_maze()
            if self.maze_step == total and self.maze_result and self.maze_result.found:
                self.lbl_maze_status.config(text="Status: ✅ Path Found!", fg=DRACULA["green"])
                self.lbl_maze_cost.config(text=f"Path Length: {self.maze_result.path_cost} steps")

    def compare_maze_algorithms(self):
        maze = GridMaze(grid=self.grid, start=self.start, goal=self.goal)
        metric_key = "manhattan" if "Manhattan" in self.maze_heur_var.get() else "euclidean"
        agents = [
            AStarAgent(name=f"A* ({metric_key.title()})", heuristic_metric=metric_key),
            BFSAgent(name="BFS"),
            DFSAgent(name="DFS"),
            GreedyAgent(name=f"Greedy ({metric_key.title()})", heuristic_metric=metric_key),
        ]
        lines = [
            f"=== Benchmark Comparison Matrix ({self.rows}x{self.cols} Maze) ===",
            f"Heuristic: {metric_key.title()}\n",
            f"{'Algorithm':<22} | {'Found?':<7} | {'Path Cost':<9} | {'Nodes':<7} | {'Time (ms)'}",
            "-" * 65,
        ]
        for ag in agents:
            res = ag.solve(maze)
            status_str = "Yes" if res.found else "No"
            cost_str = str(res.path_cost) if res.found else "N/A"
            lines.append(f"{ag.name:<22} | {status_str:<7} | {cost_str:<9} | {res.nodes_expanded:<7} | {res.execution_time_ms:.2f}")

        messagebox.showinfo("Algorithm Comparison Matrix", "\n".join(lines))

    # =====================================================
    # TAB 2: 8-Puzzle Sliding Tiles Implementation
    # =====================================================
    def _build_puzzle_tab(self):
        container = tk.Frame(self.tab_puzzle, bg=DRACULA["bg"])
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left Sidebar (Controls)
        sidebar = tk.Frame(container, bg=DRACULA["current_line"], width=320, padx=12, pady=12)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar.pack_propagate(False)

        # Right Viewport (Canvas + Status)
        viewport = tk.Frame(container, bg=DRACULA["bg"])
        viewport.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Sidebar ---
        lbl_p_preset = tk.Label(sidebar, text="Choose Initial Board Preset:", font=("Segoe UI", 9, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"])
        lbl_p_preset.pack(anchor="w", pady=(2, 2))

        self.puzzle_preset_var = tk.StringVar(value="3-Moves Easy")
        combo_p_pre = ttk.Combobox(sidebar, textvariable=self.puzzle_preset_var, values=["1-Move Trivial Solution", "3-Moves Easy", "6-Moves Medium", "10-Moves Challenging", "Scramble Random Solvable"], state="readonly")
        combo_p_pre.pack(fill=tk.X, pady=(0, 6))

        btn_apply_p = tk.Button(sidebar, text="Apply / Scramble Board", bg=DRACULA["comment"], fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.apply_puzzle_preset)
        btn_apply_p.pack(fill=tk.X, pady=(0, 10))

        # Solvability Card
        self.lbl_p_solv = tk.Label(sidebar, text="Solvability: Checking...", font=("Segoe UI", 9), justify=tk.LEFT, bg=DRACULA["dark_wall"], fg=DRACULA["green"], padx=8, pady=6)
        self.lbl_p_solv.pack(fill=tk.X, pady=(0, 10))

        lbl_p_h = tk.Label(sidebar, text="Heuristic Metric:", font=("Segoe UI", 9, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"])
        lbl_p_h.pack(anchor="w", pady=(2, 2))

        self.puzzle_heur_var = tk.StringVar(value="Manhattan Distance")
        combo_p_h = ttk.Combobox(sidebar, textvariable=self.puzzle_heur_var, values=["Manhattan Distance", "Euclidean Distance", "Misplaced Tiles"], state="readonly")
        combo_p_h.pack(fill=tk.X, pady=(0, 8))

        lbl_p_a = tk.Label(sidebar, text="Search Strategy:", font=("Segoe UI", 9, "bold"), bg=DRACULA["current_line"], fg=DRACULA["cyan"])
        lbl_p_a.pack(anchor="w", pady=(2, 2))

        self.puzzle_algo_var = tk.StringVar(value="A* Search")
        combo_p_a = ttk.Combobox(sidebar, textvariable=self.puzzle_algo_var, values=["A* Search", "Greedy Best-First", "Breadth-First Search (BFS)", "Depth-First Search (DFS)"], state="readonly")
        combo_p_a.pack(fill=tk.X, pady=(0, 10))

        # Speed Slider & Buttons
        lbl_p_sp = tk.Label(sidebar, text="⚡ Move Delay (ms):", font=("Segoe UI", 9, "bold"), bg=DRACULA["current_line"], fg=DRACULA["yellow"])
        lbl_p_sp.pack(anchor="w", pady=(2, 2))

        self.puzzle_speed_scale = tk.Scale(sidebar, from_=20, to=500, orient=tk.HORIZONTAL, bg=DRACULA["current_line"], fg=DRACULA["fg"], troughcolor=DRACULA["bg"], highlightthickness=0, activebackground=DRACULA["pink"])
        self.puzzle_speed_scale.set(180)
        self.puzzle_speed_scale.pack(fill=tk.X, pady=(0, 4))

        p_speed_row = tk.Frame(sidebar, bg=DRACULA["current_line"])
        p_speed_row.pack(fill=tk.X, pady=(0, 10))

        btn_p_slower = tk.Button(p_speed_row, text="🐢 Slower (+50ms)", bg=DRACULA["comment"], fg="white", font=("Segoe UI", 8), relief=tk.FLAT, command=lambda: self.adjust_puzzle_speed(50))
        btn_p_slower.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        btn_p_faster = tk.Button(p_speed_row, text="🚀 Faster (-50ms)", bg=DRACULA["comment"], fg="white", font=("Segoe UI", 8), relief=tk.FLAT, command=lambda: self.adjust_puzzle_speed(-50))
        btn_p_faster.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # 8-Puzzle Playback Controls
        lbl_p_ctrl = tk.Label(sidebar, text="⏯️ 1-by-1 Tile Playback Controls:", font=("Segoe UI", 9, "bold"), bg=DRACULA["current_line"], fg=DRACULA["pink"])
        lbl_p_ctrl.pack(anchor="w", pady=(4, 2))

        self.btn_puzzle_play = tk.Button(sidebar, text="▶️ Play 1-by-1 Tile Moves", bg=DRACULA["green"], fg=DRACULA["bg"], font=("Segoe UI", 10, "bold"), relief=tk.FLAT, pady=5, command=self.play_puzzle_animation, cursor="hand2")
        self.btn_puzzle_play.pack(fill=tk.X, pady=2)

        p_btn_row = tk.Frame(sidebar, bg=DRACULA["current_line"])
        p_btn_row.pack(fill=tk.X, pady=2)

        self.btn_puzzle_pause = tk.Button(p_btn_row, text="⏸️ Pause", bg=DRACULA["orange"], fg=DRACULA["bg"], font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.toggle_puzzle_pause, cursor="hand2")
        self.btn_puzzle_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.btn_puzzle_reset = tk.Button(p_btn_row, text="⏮️ Reset", bg=DRACULA["red"], fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.reset_puzzle_animation, cursor="hand2")
        self.btn_puzzle_reset.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_puzzle_step = tk.Button(p_btn_row, text="⏭️ Step", bg=DRACULA["purple"], fg=DRACULA["bg"], font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.step_puzzle_forward, cursor="hand2")
        self.btn_puzzle_step.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Move Description Card
        self.lbl_p_move_desc = tk.Label(sidebar, text="Action: Initial Board State", font=("Segoe UI", 9, "bold"), bg=DRACULA["dark_wall"], fg=DRACULA["pink"], padx=8, pady=8, wraplength=280, justify=tk.LEFT)
        self.lbl_p_move_desc.pack(fill=tk.X, pady=(12, 0))

        # --- Viewport ---
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

        # Puzzle Canvas
        self.puzzle_canvas = tk.Canvas(viewport, bg=DRACULA["bg"], highlightthickness=0)
        self.puzzle_canvas.pack(fill=tk.BOTH, expand=True)
        self.puzzle_canvas.bind("<Configure>", lambda e: self._redraw_puzzle())

        self._update_puzzle_solvability_label()
        self._redraw_puzzle()

    def adjust_puzzle_speed(self, delta: int):
        cur = self.puzzle_speed_scale.get()
        self.puzzle_speed_scale.set(max(20, min(500, cur + delta)))

    def apply_puzzle_preset(self):
        pre = self.puzzle_preset_var.get()
        if pre == "1-Move Trivial Solution":
            self.puzzle_state = (1, 2, 3, 4, 5, 6, 7, 0, 8)
        elif pre == "3-Moves Easy":
            self.puzzle_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)
        elif pre == "6-Moves Medium":
            self.puzzle_state = (1, 2, 3, 0, 4, 6, 7, 5, 8)
        elif pre == "10-Moves Challenging":
            self.puzzle_state = (1, 3, 6, 4, 2, 5, 7, 0, 8)
        else:
            while True:
                cand = list(range(9))
                random.shuffle(cand)
                cand_tuple = tuple(cand)
                if EightPuzzle.is_solvable(cand_tuple, self.puzzle_goal):
                    self.puzzle_state = cand_tuple
                    break

        self.reset_puzzle_animation()
        self._update_puzzle_solvability_label()
        self._redraw_puzzle()

    def _update_puzzle_solvability_label(self):
        inv = EightPuzzle.count_inversions(self.puzzle_state)
        g_inv = EightPuzzle.count_inversions(self.puzzle_goal)
        solv = (inv % 2) == (g_inv % 2)
        status_txt = "🟢 Solvable" if solv else "🔴 Unsolvable"
        self.lbl_p_solv.config(
            text=f"Inversion Parity Check:\n• State Inversions: {inv} ({'Even' if inv%2==0 else 'Odd'})\n• Goal Inversions: {g_inv} (Even)\n• Solvability: {status_txt}",
            fg=DRACULA["green"] if solv else DRACULA["red"],
        )

    def _redraw_puzzle(self):
        self.puzzle_canvas.delete("all")
        w = self.puzzle_canvas.winfo_width()
        h = self.puzzle_canvas.winfo_height()
        if w <= 1 or h <= 1:
            return

        board_size = min(w - 60, h - 60, 420)
        tile_size = board_size // 3
        offset_x = (w - (tile_size * 3)) // 2
        offset_y = (h - (tile_size * 3)) // 2

        curr_state = (
            self.puzzle_path[self.puzzle_step]
            if self.puzzle_path and self.puzzle_step < len(self.puzzle_path)
            else self.puzzle_state
        )

        moved_tile = None
        if self.puzzle_moves_info and self.puzzle_step < len(self.puzzle_moves_info):
            moved_tile, _ = self.puzzle_moves_info[self.puzzle_step]

        for idx, val in enumerate(curr_state):
            r, c = divmod(idx, 3)
            x1 = offset_x + c * tile_size
            y1 = offset_y + r * tile_size
            x2 = x1 + tile_size
            y2 = y1 + tile_size

            if val == 0:
                # Blank tile
                self.puzzle_canvas.create_rectangle(x1 + 4, y1 + 4, x2 - 4, y2 - 4, fill=DRACULA["dark_wall"], outline=DRACULA["comment"], width=2, dash=(4, 4))
            else:
                is_moved = (moved_tile is not None and val == moved_tile)
                fill_color = DRACULA["pink"] if is_moved else DRACULA["current_line"]
                outline_color = DRACULA["yellow"] if is_moved else DRACULA["purple"]
                text_color = DRACULA["bg"] if is_moved else DRACULA["cyan"]
                width = 3 if is_moved else 2

                self.puzzle_canvas.create_rectangle(x1 + 4, y1 + 4, x2 - 4, y2 - 4, fill=fill_color, outline=outline_color, width=width)
                self.puzzle_canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2, text=str(val), fill=text_color, font=("Segoe UI", 26, "bold"))

    def play_puzzle_animation(self):
        if self.puzzle_animating and not self.puzzle_paused:
            return

        if self.puzzle_paused:
            self.puzzle_paused = False
            self.btn_puzzle_pause.config(text="⏸️ Pause")
            self.lbl_p_status.config(text="Status: Resumed...", fg=DRACULA["yellow"])
            self._tick_puzzle_animation()
            return

        env = EightPuzzle(start_state=self.puzzle_state, goal_state=self.puzzle_goal)
        metric_key = "manhattan" if "Manhattan" in self.puzzle_heur_var.get() else ("euclidean" if "Euclidean" in self.puzzle_heur_var.get() else "misplaced")
        algo = self.puzzle_algo_var.get()

        if "A*" in algo:
            agent = AStarAgent(name=f"A* ({metric_key.title()})", heuristic_metric=metric_key)
        elif "Greedy" in algo:
            agent = GreedyAgent(name=f"Greedy ({metric_key.title()})", heuristic_metric=metric_key)
        elif "BFS" in algo:
            agent = BFSAgent(name="BFS")
        else:
            agent = DFSAgent(name="DFS")

        self.puzzle_result = agent.solve(env)
        if not self.puzzle_result.found:
            messagebox.showerror("No Solution", "This 8-puzzle configuration has no path to goal.")
            return

        self.puzzle_path = self.puzzle_result.path
        self.puzzle_moves_info = [(None, "Initial Board State")]
        for idx in range(1, len(self.puzzle_path)):
            prev_b = self.puzzle_path[idx - 1]
            curr_b = self.puzzle_path[idx]
            prev_zero = prev_b.index(0)
            curr_zero = curr_b.index(0)
            moved_t = prev_b[curr_zero]

            r0, c0 = divmod(curr_zero, 3)
            r1, c1 = divmod(prev_zero, 3)
            dr, dc = r1 - r0, c1 - c0
            direction = "UP" if dr == -1 else ("DOWN" if dr == 1 else ("LEFT" if dc == -1 else "RIGHT"))
            self.puzzle_moves_info.append((moved_t, f"Step {idx}: Tile {moved_t} slid {direction} into blank slot"))

        self.puzzle_step = 0
        self.puzzle_animating = True
        self.puzzle_paused = False
        self.btn_puzzle_pause.config(text="⏸️ Pause")
        self.lbl_p_status.config(text=f"Status: Solving ({agent.name})...", fg=DRACULA["yellow"])
        self.lbl_p_moves.config(text=f"Moves: {self.puzzle_result.path_cost}")
        self.lbl_p_nodes.config(text=f"Nodes Expanded: {self.puzzle_result.nodes_expanded}")
        self.lbl_p_time.config(text=f"Time: {self.puzzle_result.execution_time_ms:.2f} ms")

        self._tick_puzzle_animation()

    def _tick_puzzle_animation(self):
        if not self.puzzle_animating or self.puzzle_paused:
            return

        total = len(self.puzzle_path)
        if self.puzzle_step < total:
            delay = max(20, int(self.puzzle_speed_scale.get()))
            moved_t, desc = self.puzzle_moves_info[self.puzzle_step]
            self.lbl_p_move_desc.config(text=f"📍 {desc}")
            self.lbl_p_status.config(text=f"Status: Move {self.puzzle_step}/{total - 1}", fg=DRACULA["cyan"])
            self._redraw_puzzle()
            self.puzzle_step += 1
            self.root.after(delay, self._tick_puzzle_animation)
        else:
            self.puzzle_animating = False
            self.puzzle_step = total - 1
            self._redraw_puzzle()
            self.lbl_p_status.config(text="Status: ✅ Goal Board Reached!", fg=DRACULA["green"])
            self.lbl_p_move_desc.config(text="🎉 Solution Complete! Goal state matched.")

    def toggle_puzzle_pause(self):
        if not self.puzzle_animating:
            return
        self.puzzle_paused = not self.puzzle_paused
        if self.puzzle_paused:
            self.btn_puzzle_pause.config(text="▶️ Resume")
            self.lbl_p_status.config(text="Status: Paused", fg=DRACULA["yellow"])
        else:
            self.btn_puzzle_pause.config(text="⏸️ Pause")
            self.lbl_p_status.config(text="Status: Resumed...", fg=DRACULA["yellow"])
            self._tick_puzzle_animation()

    def reset_puzzle_animation(self):
        self.puzzle_animating = False
        self.puzzle_paused = False
        self.puzzle_step = 0
        self.puzzle_path.clear()
        self.puzzle_moves_info.clear()
        self.lbl_p_status.config(text="Status: Ready", fg=DRACULA["green"])
        self.lbl_p_moves.config(text="Moves: --")
        self.lbl_p_nodes.config(text="Nodes Expanded: 0")
        self.lbl_p_time.config(text="Time: 0.0 ms")
        self.lbl_p_move_desc.config(text="Action: Initial Board State")
        self.btn_puzzle_pause.config(text="⏸️ Pause")
        self._redraw_puzzle()

    def step_puzzle_forward(self):
        if not self.puzzle_path:
            return
        total = len(self.puzzle_path)
        if self.puzzle_step < total - 1:
            self.puzzle_step += 1
            moved_t, desc = self.puzzle_moves_info[self.puzzle_step]
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
        print("Note: On Linux, ensure a graphical environment ($DISPLAY) is active.")
        print("If on Ubuntu/Debian, install Tkinter via: sudo apt install python3-tk")
        print("Alternatively, launch the web GUI: uv run streamlit run app.py\n")
        sys.exit(1)

    app = MazeDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
