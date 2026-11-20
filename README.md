# MazeMind AI — Intelligent Puzzle Solver & Search Algorithm Visualizer

[![Python 3.12](https://img.shields.io/badge/Python-3.12-green.svg)](https://python.org)
[![Package Manager: uv](https://img.shields.io/badge/Environment-uv-indigo.svg)](https://github.com/astral-sh/uv)
[![Theme: Dracula](https://img.shields.io/badge/Theme-Dracula%20Spec-BD93F9.svg?logo=dracula&logoColor=white)](https://draculatheme.com/spec)
[![Platform: Linux | Windows | macOS](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-orange.svg)](https://github.com/QZainSami/mazemind-ai)
[![Streamlit Cloud Ready](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-red.svg)](https://share.streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview

**MazeMind AI** is an interactive artificial intelligence platform designed to solve, animate, and compare classical state-space search algorithms across two benchmark problem domains: **2D Grid Mazes** and the **8-Puzzle Sliding Tiles**.

Built entirely from scratch in pure Python using modular object-oriented design, the system relies on zero external AI APIs or black-box packages. It is delivered in two complementary forms:
* 🌐 **Interactive Web Application (`app.py`):** Built with Streamlit for cloud deployment on Streamlit Community Cloud.
* 🖥️ **Native Desktop Canvas GUI (`gui_app.py`):** High-performance Tkinter Canvas providing smooth, 60 FPS local step-by-step animations.

Both applications share the exact same underlying environment models, autonomous search agents, and visual design.

---

## 2. Key Capabilities & Highlights

* **Pure Python Implementation:** Breadth-First Search (BFS), Depth-First Search (DFS), Greedy Best-First Search, and A\* Search implemented from foundational data structures (`collections.deque`, explicit LIFO stacks, `heapq` priority queues).
* **Dual Problem Domains:**
  * **2D Grid Maze Solver:** Interactive obstacle density sliders, custom wall painting/toggling, and real-time visualization of explored nodes, active frontiers, and optimal paths.
  * **8-Puzzle Sliding Tiles:** Randomized solvable boards, custom configuration inputs, and live state-space search.
* **1-by-1 Animated Tile Sliding:** The 8-puzzle solver visually animates tiles moving one by one into the empty space, highlighting the moving tile in pink with real-time directional labels (e.g., `Tile 6 slid UP into blank slot`).
* **Universal Playback Controls:** Full control over animation playback across both web and desktop interfaces:
  * ▶️ **Play / ⏸️ Pause:** Start and freeze the search visualization at any intermediate step.
  * 🔄 **Reset:** Instantly return the board or maze to its initial state.
  * ⏭️ **Step Forward:** Advance the exploration step-by-step to inspect search decisions.
  * 🐢 **Slower / 🚀 Faster:** Quick preset buttons to adjust playback speed.
  * ⏱️ **Speed Slider:** Continuous speed regulation (20 ms to 500 ms per step).
* **Mathematical Solvability Guarantees:** 8-puzzle configurations are verified using the **Inversion Parity Theorem** to guarantee that only mathematically solvable configurations are executed.
* **Multi-Algorithm Benchmarking:** Directly compare BFS, DFS, Greedy Best-First, and A\* across **path cost**, **nodes expanded**, **max frontier size**, and **execution time (ms)** with comparative charts.

---

## 3. Supported Search Algorithms & Heuristics

| Algorithm | Category | Data Structure | Optimality Guarantee | Completeness |
|:---|:---:|:---:|:---:|:---:|
| **Breadth-First Search (BFS)** | Uninformed | FIFO Queue (`deque`) | Optimal for uniform step costs | Complete |
| **Depth-First Search (DFS)** | Uninformed | LIFO Stack (explicit) | Non-optimal | Complete in finite spaces |
| **Greedy Best-First Search** | Informed | Priority Queue (`heapq`) | Non-optimal ($f(n) = h(n)$) | Complete in finite spaces |
| **A\* Search** | Informed | Priority Queue (`heapq`) | Optimal with admissible heuristic | Complete |

### Heuristic Functions
* **Manhattan Distance ($L_1$ Norm):** Calculates grid distance ignoring obstacles:
  $$h_{\text{Manhattan}}(n) = |r_n - r_{\text{goal}}| + |c_n - c_{\text{goal}}|$$
* **Euclidean Distance ($L_2$ Norm):** Calculates straight-line geometric distance:
  $$h_{\text{Euclidean}}(n) = \sqrt{(r_n - r_{\text{goal}})^2 + (c_n - c_{\text{goal}})^2}$$
* **Dominance Property:** Because diagonal steps are disallowed on orthogonal grids, $h_{\text{Manhattan}}(n) \ge h_{\text{Euclidean}}(n)$ for all states. Manhattan distance dominates Euclidean distance and expands fewer or equal nodes under A\*.

---

## 4. 8-Puzzle Solvability (Inversion Parity Theorem)

An inversion occurs whenever a higher-numbered tile precedes a lower-numbered tile in the linear sequence of the board (excluding the blank tile $0$):

$$\mathrm{Inv}(S) = \sum_{1 \le i < j \le 8} [S_i > S_j \land S_i \ne 0 \land S_j \ne 0]$$

On a $3 \times 3$ board (odd width $W = 3$):
* Horizontal tile slides do not change the linear order of tiles ($\Delta \mathrm{Inv} = 0$).
* Vertical tile slides shift a tile past exactly two other tiles ($\Delta \mathrm{Inv} \in \{-2, 0, +2\}$).

Since every valid slide changes the inversion count by an even number, the parity of inversions remains strictly invariant:

$$\mathrm{Solvable}(S, G) \iff \mathrm{Inv}(S) \equiv \mathrm{Inv}(G) \pmod 2$$

The application computes $\mathrm{Inv}(S)$ on every board generation to ensure users never run an unsolvable puzzle.

---

## 5. Repository Directory Structure

```text
mazemind-ai/
├── .streamlit/
│   └── config.toml                 # Dracula theme configuration for Streamlit
├── pyproject.toml                  # uv project configuration and dependencies
├── uv.lock                         # Deterministic uv lockfile
├── requirements.txt                # Exported via uv export for Streamlit Cloud
├── README.md                       # Complete documentation and setup guides
├── LICENSE                         # MIT License
├── app.py                          # Streamlit web application with live animated solving
├── gui_app.py                      # Standalone native desktop Canvas GUI (60 FPS)
├── src/
│   └── mazemind/
│       ├── __init__.py             # Public module exports
│       ├── environments.py         # OOP classes: Environment, GridMaze, EightPuzzle
│       ├── search_agents.py        # OOP classes: SearchAgent, BFSAgent, DFSAgent, GreedyAgent, AStarAgent
│       └── visualizer.py           # SearchVisualizer (1-by-1 tile sliding, comparison charts)
└── tests/
    └── test_search.py              # Unit tests for BFS, DFS, Greedy, A*, heuristics, and parity
```

---

## 6. Local Setup & Execution Guide

### 🐧 Linux (Ubuntu, Debian, Fedora, Arch, WSL)

#### 1. Install System Prerequisites
Install Python 3.12 and Tkinter (for the desktop GUI):
* **Ubuntu / Debian / WSL:**
  ```bash
  sudo apt update
  sudo apt install -y python3 python3-pip python3-tk curl git
  ```
* **Fedora / RHEL:**
  ```bash
  sudo dnf install -y python3 python3-pip python3-tkinter curl git
  ```
* **Arch Linux:**
  ```bash
  sudo pacman -S python python-pip tk curl git
  ```

#### 2. Install `uv` Package Manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # reload environment
```

#### 3. Clone & Sync Environment
```bash
git clone https://github.com/QZainSami/mazemind-ai.git
cd mazemind-ai

# Synchronize virtual environment deterministically
uv sync
```

#### 4. Run Unit Test Suite
```bash
uv run pytest
```

#### 5. Launch the Applications
* **Streamlit Web Application:**
  ```bash
  uv run streamlit run app.py
  ```
  Open your browser at `http://localhost:8501`.

* **Native Desktop Canvas GUI:**
  ```bash
  uv run python gui_app.py
  ```

---

### 🪟 Windows (PowerShell / Command Prompt)

#### 1. Install `uv`
In PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. Clone & Sync
```powershell
git clone https://github.com/QZainSami/mazemind-ai.git
cd mazemind-ai
uv sync
```

#### 3. Run Unit Tests
```powershell
uv run pytest
```

#### 4. Launch Applications
* **Streamlit Web Application:**
  ```powershell
  uv run streamlit run app.py
  ```
* **Native Desktop Canvas GUI:**
  ```powershell
  uv run python gui_app.py
  ```

---

## 7. Streamlit Community Cloud Deployment

To host this project online on [Streamlit Community Cloud](https://share.streamlit.io):

1. Commit and push your code to GitHub:
   ```bash
   git add .
   git commit -m "feat: complete MazeMind AI implementation"
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **Create app** > **Deploy a public app from GitHub**.
4. Set Repository: `QZainSami/mazemind-ai`.
5. Set Main file path: `app.py`.
6. Click **Deploy!** — Streamlit Cloud automatically detects `requirements.txt` and deploys your application within seconds.

---

## 8. Credits & Theme

* **Visual Theme:** Styled with the official [Dracula Theme](https://draculatheme.com/spec) for high-contrast, comfortable visual aesthetics across both web and desktop interfaces.
* **License:** Released under the [MIT License](LICENSE).
