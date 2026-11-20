"""
MazeMind AI — Native Desktop GUI Application.
High-performance Canvas-based GUI featuring:
- Real-time 60 FPS animated search with dynamic speed control (1ms to 150ms)
- Interactive grid editing (Left-click to place walls, Right-click to erase)
- Heuristic selection: Manhattan Distance & Euclidean Distance
- Both 2D Grid Maze and 8-Puzzle sliding tile solvers
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


class MazeDesktopApp:
    """
    Native Desktop GUI for MazeMind AI with interactive drawing and real-time animation.
    """

    COLOR_BG = "#1E1E2E"
    COLOR_PANEL = "#252538"
    COLOR_GRID_BG = "#FFFFFF"
    COLOR_WALL = "#2B2D42"
    COLOR_START = "#2ECC71"
    COLOR_GOAL = "#E74C3C"
    COLOR_VISITED = "#74B9FF"
    COLOR_PATH = "#FF7675"
    COLOR_PATH_LINE = "#D63031"
    COLOR_TEXT = "#ECEFF1"
    COLOR_ACCENT = "#6C5CE7"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MazeMind AI — Puzzle Solver & Search Visualizer (CSL-325)")
        self.root.geometry("1100x750")
        self.root.minsize(980, 680)
        self.root.configure(bg=self.COLOR_BG)

        # Maze parameters
        self.rows = 15
        self.cols = 15
        self.cell_size = 32
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int32)

        # Animation state
        self.is_animating = False
        self.is_paused = False
        self.anim_step_index = 0
        self.anim_history: List[Tuple[int, int]] = []
        self.anim_path: List[Tuple[int, int]] = []
        self.current_result: Optional[SearchResult] = None

        # Build UI layout
        self._setup_styles()
        self._build_widgets()
        self._redraw_grid()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background=self.COLOR_PANEL, foreground=self.COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#A29BFE")
        style.configure("TFrame", background=self.COLOR_PANEL)
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=5)
        style.configure("Accent.TButton", background=self.COLOR_ACCENT, foreground="white", font=("Segoe UI", 10, "bold"))

    def _build_widgets(self):
        self.main_container = tk.Frame(self.root, bg=self.COLOR_BG)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Left Control Sidebar
        self.sidebar = tk.Frame(self.main_container, bg=self.COLOR_PANEL, width=320, padx=14, pady=14)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.sidebar.pack_propagate(False)

        # Right Viewport (Canvas + Status bar)
        self.viewport = tk.Frame(self.main_container, bg=self.COLOR_BG)
        self.viewport.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_sidebar()
        self._build_viewport()

    def _build_sidebar(self):
        # Title
        title_lbl = tk.Label(
            self.sidebar,
            text="MazeMind AI",
            font=("Segoe UI", 18, "bold"),
            bg=self.COLOR_PANEL,
            fg="#A29BFE",
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            self.sidebar,
            text="CSL-325 • Real-Time Search Visualizer",
            font=("Segoe UI", 9),
            bg=self.COLOR_PANEL,
            fg="#B2BEC3",
        )
        sub_lbl.pack(anchor="w", pady=(0, 8))

        # Academic Profile Card
        team_lbl = tk.Label(
            self.sidebar,
            text="Group: Samiullah, Musavir, Farman\nInstructor: Sir Shahid | BS (CS) 5-B",
            font=("Segoe UI", 8),
            justify=tk.LEFT,
            bg="#2D3250",
            fg="#F9F9F9",
            padx=8,
            pady=4,
            relief=tk.RIDGE,
        )
        team_lbl.pack(fill=tk.X, pady=(0, 10))

        # --- Section 1: Heuristic Selector ---
        lbl_heur = tk.Label(
            self.sidebar,
            text="Heuristic Distance Function:",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT,
        )
        lbl_heur.pack(anchor="w", pady=(4, 2))

        self.heur_var = tk.StringVar(value="Manhattan Distance")
        self.combo_heur = ttk.Combobox(
            self.sidebar,
            textvariable=self.heur_var,
            values=["Manhattan Distance", "Euclidean Distance"],
            state="readonly",
            font=("Segoe UI", 9),
        )
        self.combo_heur.pack(fill=tk.X, pady=(0, 8))

        # --- Section 2: Algorithm Selector ---
        lbl_algo = tk.Label(
            self.sidebar,
            text="Search Algorithm:",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT,
        )
        lbl_algo.pack(anchor="w", pady=(4, 2))

        self.algo_var = tk.StringVar(value="A* Search")
        algos = ["A* Search", "Breadth-First Search (BFS)", "Depth-First Search (DFS)", "Greedy Best-First Search"]
        self.combo_algo = ttk.Combobox(
            self.sidebar,
            textvariable=self.algo_var,
            values=algos,
            state="readonly",
            font=("Segoe UI", 9),
        )
        self.combo_algo.pack(fill=tk.X, pady=(0, 8))

        # --- Section 3: Animation Speed Slider ---
        lbl_speed = tk.Label(
            self.sidebar,
            text="Animation Delay (1 ms to 150 ms):",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT,
        )
        lbl_speed.pack(anchor="w", pady=(4, 2))

        self.speed_scale = tk.Scale(
            self.sidebar,
            from_=1,
            to=150,
            orient=tk.HORIZONTAL,
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT,
            troughcolor="#2D3436",
            highlightthickness=0,
            activebackground=self.COLOR_ACCENT,
        )
        self.speed_scale.set(15)
        self.speed_scale.pack(fill=tk.X, pady=(0, 10))

        # --- Section 4: Action Buttons ---
        self.btn_solve = tk.Button(
            self.sidebar,
            text="▶️ Solve & Animate",
            bg="#00B894",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=6,
            command=self.start_solving_animation,
            cursor="hand2",
        )
        self.btn_solve.pack(fill=tk.X, pady=(4, 4))

        btn_row1 = tk.Frame(self.sidebar, bg=self.COLOR_PANEL)
        btn_row1.pack(fill=tk.X, pady=2)

        self.btn_pause = tk.Button(
            btn_row1,
            text="⏸️ Pause",
            bg="#636E72",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            command=self.toggle_pause,
            cursor="hand2",
        )
        self.btn_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.btn_step = tk.Button(
            btn_row1,
            text="⏭️ Step",
            bg="#0984E3",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            command=self.step_forward,
            cursor="hand2",
        )
        self.btn_step.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        btn_row2 = tk.Frame(self.sidebar, bg=self.COLOR_PANEL)
        btn_row2.pack(fill=tk.X, pady=(6, 2))

        self.btn_random = tk.Button(
            btn_row2,
            text="🎲 Random Maze",
            bg="#6C5CE7",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            command=self.generate_random_maze,
            cursor="hand2",
        )
        self.btn_random.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.btn_clear = tk.Button(
            btn_row2,
            text="🧹 Clear Walls",
            bg="#D63031",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            command=self.clear_walls,
            cursor="hand2",
        )
        self.btn_clear.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Compare All Solvers Button
        self.btn_compare = tk.Button(
            self.sidebar,
            text="📊 Benchmark All 4 Algorithms",
            bg="#E17055",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            command=self.compare_all_algorithms,
            cursor="hand2",
        )
        self.btn_compare.pack(fill=tk.X, pady=(6, 8))

        # Instructions Label
        info_text = (
            "💡 Interaction Guide:\n"
            "• Left Click / Drag: Draw Walls\n"
            "• Right Click / Drag: Erase Walls\n"
            "• Adjust Speed slider anytime during animation"
        )
        lbl_info = tk.Label(
            self.sidebar,
            text=info_text,
            font=("Segoe UI", 8),
            justify=tk.LEFT,
            bg=self.COLOR_PANEL,
            fg="#B2BEC3",
        )
        lbl_info.pack(anchor="w", pady=(8, 0))

    def _build_viewport(self):
        # Top KPI bar
        self.kpi_frame = tk.Frame(self.viewport, bg=self.COLOR_PANEL, padx=12, pady=8)
        self.kpi_frame.pack(fill=tk.X, pady=(0, 8))

        self.lbl_status = tk.Label(
            self.kpi_frame,
            text="Status: Ready",
            font=("Segoe UI", 10, "bold"),
            bg=self.COLOR_PANEL,
            fg="#55EFC4",
        )
        self.lbl_status.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_cost = tk.Label(
            self.kpi_frame,
            text="Path Length: --",
            font=("Segoe UI", 10),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT,
        )
        self.lbl_cost.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_nodes = tk.Label(
            self.kpi_frame,
            text="Nodes Visited: 0",
            font=("Segoe UI", 10),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT,
        )
        self.lbl_nodes.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_time = tk.Label(
            self.kpi_frame,
            text="Time: 0.0 ms",
            font=("Segoe UI", 10),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT,
        )
        self.lbl_time.pack(side=tk.LEFT)

        # Canvas Frame
        self.canvas_frame = tk.Frame(self.viewport, bg=self.COLOR_BG)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg=self.COLOR_WALL,
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Mouse event bindings for wall drawing
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<B1-Motion>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<B3-Motion>", self._on_right_click)
        self.canvas.bind("<Configure>", lambda e: self._redraw_grid())

    # --- Mouse Drawing Handlers ---
    def _coords_to_cell(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        grid_w = self.cols * self.cell_size
        grid_h = self.rows * self.cell_size
        offset_x = max(0, (w - grid_w) // 2)
        offset_y = max(0, (h - grid_h) // 2)

        c = (x - offset_x) // self.cell_size
        r = (y - offset_y) // self.cell_size

        if 0 <= r < self.rows and 0 <= c < self.cols:
            return int(r), int(c)
        return None

    def _on_left_click(self, event):
        if self.is_animating:
            return
        cell = self._coords_to_cell(event.x, event.y)
        if cell and cell != self.start and cell != self.goal:
            r, c = cell
            self.grid[r, c] = 1
            self._redraw_grid()

    def _on_right_click(self, event):
        if self.is_animating:
            return
        cell = self._coords_to_cell(event.x, event.y)
        if cell and cell != self.start and cell != self.goal:
            r, c = cell
            self.grid[r, c] = 0
            self._redraw_grid()

    # --- Grid Drawing Logic ---
    def _redraw_grid(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

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

        visited_set = set(self.anim_history[: self.anim_step_index])
        path_set = set(self.anim_path) if self.anim_step_index >= len(self.anim_history) and self.anim_path else set()

        for r in range(self.rows):
            for c in range(self.cols):
                x1 = offset_x + c * self.cell_size
                y1 = offset_y + r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if (r, c) == self.start:
                    color = self.COLOR_START
                elif (r, c) == self.goal:
                    color = self.COLOR_GOAL
                elif (r, c) in path_set:
                    color = self.COLOR_PATH
                elif (r, c) in visited_set:
                    color = self.COLOR_VISITED
                elif self.grid[r, c] == 1:
                    color = self.COLOR_WALL
                else:
                    color = self.COLOR_GRID_BG

                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline="#DFE6E9", width=1
                )

                if (r, c) == self.start:
                    self.canvas.create_text(
                        (x1 + x2) // 2, (y1 + y2) // 2, text="S", fill="white", font=("Segoe UI", 12, "bold")
                    )
                elif (r, c) == self.goal:
                    self.canvas.create_text(
                        (x1 + x2) // 2, (y1 + y2) // 2, text="G", fill="white", font=("Segoe UI", 12, "bold")
                    )

        if path_set and len(self.anim_path) > 1:
            points = []
            for r, c in self.anim_path:
                px = offset_x + c * self.cell_size + self.cell_size // 2
                py = offset_y + r * self.cell_size + self.cell_size // 2
                points.extend([px, py])
            self.canvas.create_line(points, fill=self.COLOR_PATH_LINE, width=3, smooth=True)

    # --- Maze Actions ---
    def generate_random_maze(self):
        if self.is_animating:
            return
        rng = np.random.default_rng()
        self.grid = (rng.random((self.rows, self.cols)) < 0.24).astype(np.int32)
        self.grid[self.start[0], self.start[1]] = 0
        self.grid[self.goal[0], self.goal[1]] = 0
        self.anim_history.clear()
        self.anim_path.clear()
        self.anim_step_index = 0
        self.lbl_status.config(text="Status: Random Maze Generated", fg="#55EFC4")
        self.lbl_cost.config(text="Path Length: --")
        self.lbl_nodes.config(text="Nodes Visited: 0")
        self.lbl_time.config(text="Time: 0.0 ms")
        self._redraw_grid()

    def clear_walls(self):
        if self.is_animating:
            return
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.anim_history.clear()
        self.anim_path.clear()
        self.anim_step_index = 0
        self.lbl_status.config(text="Status: All Walls Cleared", fg="#55EFC4")
        self._redraw_grid()

    # --- Animation & Solving Loop ---
    def start_solving_animation(self):
        if self.is_animating:
            return

        maze = GridMaze(grid=self.grid, start=self.start, goal=self.goal)
        algo_name = self.algo_var.get()
        metric_key = "manhattan" if "Manhattan" in self.heur_var.get() else "euclidean"

        if "A*" in algo_name:
            agent = AStarAgent(name=f"A* ({metric_key.title()})", heuristic_metric=metric_key)
        elif "BFS" in algo_name:
            agent = BFSAgent(name="BFS")
        elif "DFS" in algo_name:
            agent = DFSAgent(name="DFS")
        else:
            agent = GreedyAgent(name=f"Greedy ({metric_key.title()})", heuristic_metric=metric_key)

        self.current_result = agent.solve(maze)
        self.anim_history = self.current_result.exploration_history
        self.anim_path = self.current_result.path if self.current_result.found else []
        self.anim_step_index = 0

        self.is_animating = True
        self.is_paused = False
        self.btn_pause.config(text="⏸️ Pause")
        self.lbl_status.config(text=f"Status: Searching with {agent.name}...", fg="#FDCB6E")
        self.lbl_time.config(text=f"Time: {self.current_result.execution_time_ms:.2f} ms")

        self._run_animation_tick()

    def _run_animation_tick(self):
        if not self.is_animating or self.is_paused:
            return

        total_steps = len(self.anim_history)

        if self.anim_step_index < total_steps:
            delay = max(1, int(self.speed_scale.get()))
            stride = 1 if delay > 30 else (2 if delay > 10 else 4)

            self.anim_step_index = min(total_steps, self.anim_step_index + stride)
            self.lbl_nodes.config(text=f"Nodes Visited: {self.anim_step_index}/{total_steps}")
            self._redraw_grid()
            self.root.after(delay, self._run_animation_tick)
        else:
            self.is_animating = False
            self._redraw_grid()
            if self.current_result and self.current_result.found:
                self.lbl_status.config(text="Status: ✅ Path Found!", fg="#00B894")
                self.lbl_cost.config(text=f"Path Length: {self.current_result.path_cost} steps")
            else:
                self.lbl_status.config(text="Status: ❌ Unsolvable Maze!", fg="#D63031")
                self.lbl_cost.config(text="Path Length: N/A")

    def toggle_pause(self):
        if not self.is_animating:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="▶️ Resume")
            self.lbl_status.config(text="Status: Paused", fg="#FFEAA7")
        else:
            self.btn_pause.config(text="⏸️ Pause")
            self.lbl_status.config(text="Status: Resumed...", fg="#FDCB6E")
            self._run_animation_tick()

    def step_forward(self):
        if not self.anim_history:
            return
        total_steps = len(self.anim_history)
        if self.anim_step_index < total_steps:
            self.anim_step_index = min(total_steps, self.anim_step_index + 1)
            self.lbl_nodes.config(text=f"Nodes Visited: {self.anim_step_index}/{total_steps}")
            self._redraw_grid()
            if self.anim_step_index == total_steps and self.current_result and self.current_result.found:
                self.lbl_status.config(text="Status: ✅ Path Found!", fg="#00B894")
                self.lbl_cost.config(text=f"Path Length: {self.current_result.path_cost} steps")

    def compare_all_algorithms(self):
        """Runs all 4 algorithms and displays comparative metrics in a popup dialog."""
        maze = GridMaze(grid=self.grid, start=self.start, goal=self.goal)
        metric_key = "manhattan" if "Manhattan" in self.heur_var.get() else "euclidean"

        agents = [
            AStarAgent(name=f"A* ({metric_key.title()})", heuristic_metric=metric_key),
            BFSAgent(name="BFS"),
            DFSAgent(name="DFS"),
            GreedyAgent(name=f"Greedy ({metric_key.title()})", heuristic_metric=metric_key),
        ]

        summary_lines = [
            f"=== Benchmark Comparison Matrix ({self.rows}x{self.cols} Maze) ===",
            f"Heuristic: {metric_key.title()}\n",
            f"{'Algorithm':<22} | {'Found?':<7} | {'Path Cost':<9} | {'Nodes':<7} | {'Time (ms)'}",
            "-" * 65,
        ]

        for ag in agents:
            res = ag.solve(maze)
            status_str = "Yes" if res.found else "No"
            cost_str = str(res.path_cost) if res.found else "N/A"
            summary_lines.append(
                f"{ag.name:<22} | {status_str:<7} | {cost_str:<9} | {res.nodes_expanded:<7} | {res.execution_time_ms:.2f}"
            )

        messagebox.showinfo("Algorithm Comparison", "\n".join(summary_lines))


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
