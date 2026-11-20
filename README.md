# MazeMind AI — Intelligent Puzzle Solver & Search Algorithm Visualizer

[![Course: CSL-325](https://img.shields.io/badge/Course-CSL--325%20AI%20Lab-blue.svg)](https://github.com/QZainSami/mazemind-ai)
[![Semester: Fall 2026](https://img.shields.io/badge/Semester-Fall%202026%20%7C%20BS%20%28CS%29%205--B-purple.svg)](https://github.com/QZainSami/mazemind-ai)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green.svg)](https://python.org)
[![Package Manager: uv](https://img.shields.io/badge/Environment-uv-indigo.svg)](https://github.com/astral-sh/uv)
[![Theme: Dracula](https://img.shields.io/badge/Theme-Dracula%20Spec-BD93F9.svg?logo=dracula&logoColor=white)](https://draculatheme.com/spec)
[![Platform: Linux | Windows | macOS](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-orange.svg)](https://github.com/QZainSami/mazemind-ai)
[![Streamlit Cloud Ready](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-red.svg)](https://share.streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview & Team Profile

* **Course:** CSL-325 Artificial Intelligence Lab (Fall 2026)
* **Class:** BS (CS) 5 - B
* **Course Instructor:** Sir Shahid

### Team Members & Project Roles
| S# | Student Name | Registration No. | Email | Project Contributions |
|:---:|:---|:---:|:---|:---|
| **1** | **Samiullah Qureshi** | `02-134242-111` | qsamiullah90@gmail.com | System architecture, OOP design, informed search algorithms (Greedy Best-First, A*), and Streamlit Cloud deployment. |
| **2** | **Muhammad Musavir Waseem** | `02-134242-008` | musavirrizvi349@gmail.com | Uninformed search algorithms (BFS, DFS), 8-puzzle state-space logic and solvability validation, and interactive grid UI. |
| **3** | **Farman Ali** | `02-134241-015` | farmanalichhajro@gmail.com | Search step visualization, performance comparison tables/charts, unit testing, and project documentation. |

---

## 2. Executive Summary of Project Proposal

**MazeMind AI** is a purely software-based artificial intelligence application designed to solve and analyze grid-based mazes and 8-puzzle problems. Built in Python using object-oriented programming and deployed as both an interactive web application on **Streamlit Community Cloud** and a high-performance **native desktop Canvas GUI**, the system implements uninformed search algorithms (**Breadth-First Search** and **Depth-First Search**) and informed heuristic search algorithms (**Greedy Best-First Search** and **A\* Search**) completely from scratch without relying on pre-trained models or external APIs.

Users can interactively design custom maze layouts, adjust obstacle density, or generate randomized 8-puzzle boards, and observe how each search algorithm explores states step-by-step. The application visually compares explored nodes, frontier growth, path length, and execution time to demonstrate the behavior and trade-offs of classical AI search techniques.

---

## 3. Scope & Objectives

### Scope (Included vs. Excluded)
* **Included:**
  * Object-oriented 2D grid maze and 8-puzzle environments.
  * Custom Python implementations of BFS, DFS, Greedy Best-First Search, and A* Search (supporting both **Manhattan** and **Euclidean** heuristics).
  * Interactive grid editor and 8-puzzle randomizer with mathematical solvability checking (Inversion Parity Theorem).
  * Step-by-step 1-by-1 sliding tile animation for 8-puzzle and path discovery for 2D mazes with full playback controls (Play, Pause, Reset, Step, Speed).
  * Dark mode design adhering to the official **Dracula Theme Specification**.
  * Side-by-side performance comparison tables and comparative bar charts.
  * Dual delivery: **Streamlit Web Application** (`app.py`) for cloud hosting and **Native Desktop GUI** (`gui_app.py`) for 60 FPS local canvas animation.
* **Excluded:**
  * Physical hardware devices or sensors.
  * 3D game engines.
  * Pre-trained machine learning models or external third-party AI APIs.

### Project Objectives
1. Build a modular, object-oriented Python architecture modeling 2D grid mazes, 8-puzzle boards, and autonomous search agents.
2. Implement Breadth-First Search (BFS), Depth-First Search (DFS), Greedy Best-First Search, and A* Search from scratch.
3. Support admissible heuristics (Manhattan distance and Euclidean distance) for informed search and solvability validation for 8-puzzle configurations.
4. Animate step-by-step 1-by-1 tile movements, node exploration, frontier states, and final solution paths inside an interactive user interface.
5. Provide comprehensive playback controls (Play, Pause, Reset, Step, Slower, Faster, Speed sliders).
6. Compare algorithm performance across path cost, total nodes expanded, max frontier size, and execution time (ms).
7. Deploy the complete software application publicly on Streamlit Community Cloud.

---

## 4. Visual Design & Dracula Theme Specification

MazeMind AI adopts the official [Dracula Theme Specification](https://draculatheme.com/spec) across both the web application (`app.py`) and native desktop GUI (`gui_app.py`) for maximum legibility, eye comfort, and aesthetic consistency:

| Dracula Element | Hex Code | UI / Visualization Application |
|:---|:---:|:---|
| **Background** | `#282a36` | Main window canvas, background plots, dark container panels |
| **Current Line / Selection** | `#44475a` | Grid gridlines, unvisited maze cells, inactive tile backgrounds |
| **Foreground** | `#f8f8f2` | Primary text, tile numbers, axes labels, table content |
| **Comment** | `#6272a4` | Subtitles, muted helper captions, board borders |
| **Purple** | `#bd93f9` | Accent headers, solution paths, primary CTA buttons |
| **Pink** | `#ff79c6` | **Active 1-by-1 sliding tile** in 8-puzzle, frontier nodes |
| **Green** | `#50fa7b` | Start cell `(0,0)`, success status badges, optimal metrics |
| **Red** | `#ff5555` | Goal cell, maze wall obstacles, unsolvable warnings |
| **Cyan** | `#8be9fd` | Explored nodes, visited grid cells, secondary highlights |
| **Yellow** | `#f1fa8c` | Blank tile `[0]` slot, speed accents, warning notices |
| **Orange** | `#ffb86c` | Inversion counts, heuristic weights, runtime metric indicators |

---

## 5. Interactive Animation & Playback Controls

Both the Streamlit Web Application and Desktop GUI feature comprehensive, real-time animation engines:

### 🧩 1-by-1 Animated 8-Puzzle Tile Sliding
* **Sliding Mechanics:** Rather than abruptly jumping between puzzle states, the visualizer calculates the delta between consecutive states $S_{t-1}$ and $S_t$.
* **Active Tile Highlighting:** The exact numbered tile moving into the empty slot is highlighted in **Dracula Pink (`#ff79c6`)**.
* **Action Indicator:** Displays real-time move descriptions (e.g. `📍 Step 4: Tile 6 slid UP into blank slot`).
* **Move History Log:** An expandable step-by-step trajectory recording each intermediate board state and slide direction.

### 🎮 Universal Playback Controls
* **▶️ Play / ⏸️ Pause:** Seamlessly start and freeze the animation at any exploration or solution step.
* **🔄 Reset:** Instantly rewind the environment back to the initial start state.
* **⏭️ Step Forward:** Advance the solver one step at a time for deep analysis of heuristic decisions.
* **🐢 Slower / 🚀 Faster:** Quick preset buttons to decrease or increase animation delay.
* **⏱️ Speed Slider:** Continuous fine-grained speed control (ranging from 20 ms to 500 ms per step).

---

## 6. Theoretical Foundations & Mathematical Formulations

### A. Classical Search Algorithms
1. **Breadth-First Search (BFS):** Explores shallowest unexpanded nodes first using a FIFO queue (`collections.deque`). Guarantees optimal (shortest) path cost for uniform step costs:
   $$g(n) = \text{depth}(n)$$
2. **Depth-First Search (DFS):** Explores deepest unexpanded nodes first using an explicit LIFO stack with strict cycle prevention. Memory-efficient ($O(b \cdot m)$ space).
3. **Greedy Best-First Search:** Evaluates nodes solely by heuristic estimate to goal:
   $$f(n) = h(n)$$
4. **A\* Search:** Combines historical cost $g(n)$ and heuristic estimate $h(n)$ to guarantee path optimality when $h(n)$ is admissible:
   $$f(n) = g(n) + h(n)$$

### B. Admissible Heuristics
* **Manhattan Distance ($L_1$ Norm):** Represents distance on a 4-directional grid without obstacles:
   $$h_{\text{Manhattan}}(n) = |r_n - r_{\text{goal}}| + |c_n - c_{\text{goal}}|$$
* **Euclidean Distance ($L_2$ Norm):** Straight-line Euclidean distance:
   $$h_{\text{Euclidean}}(n) = \sqrt{(r_n - r_{\text{goal}})^2 + (c_n - c_{\text{goal}})^2}$$
* **Admissibility & Dominance:** Since diagonal moves are disallowed on grid topologies, $h_{\text{Manhattan}}(n) \ge h_{\text{Euclidean}}(n)$ everywhere. Manhattan distance dominates Euclidean distance and expands fewer or equal nodes in A*.

### C. 8-Puzzle Solvability (Inversion Parity Theorem)
An inversion occurs when a tile $S_i > S_j$ for indices $i < j$ (ignoring the blank tile $0$):
$$\mathrm{Inv}(S) = \sum_{1 \le i < j \le 8} [S_i > S_j \land S_i \ne 0 \land S_j \ne 0]$$

On a $3 \times 3$ board (odd width $W = 3$), horizontal tile slides preserve the linear order of tiles ($\Delta \mathrm{Inv} = 0$), while vertical slides swap a tile across two other tiles ($\Delta \mathrm{Inv} \in \{-2, 0, +2\}$). Therefore, inversion parity is an invariant:
$$\mathrm{Solvable}(S, G) \iff \mathrm{Inv}(S) \equiv \mathrm{Inv}(G) \pmod 2$$

---

## 7. Repository Directory Structure

```text
mazemind-ai/
├── .streamlit/
│   └── config.toml                 # Dracula theme configuration for Streamlit
├── pyproject.toml                  # uv project configuration and dependencies
├── uv.lock                         # Deterministic uv lockfile
├── requirements.txt                # Exported via uv export for Streamlit Cloud
├── README.md                       # Complete documentation, proposal details, and guides
├── LICENSE                         # MIT License (Sami, 2026)
├── app.py                          # Streamlit web application with live animated solving
├── gui_app.py                      # Standalone native desktop Canvas GUI (60 FPS)
├── src/
│   └── mazemind/
│       ├── __init__.py             # Public module exports
│       ├── environments.py         # OOP classes: Environment, GridMaze, EightPuzzle
│       ├── search_agents.py        # OOP classes: SearchAgent, BFSAgent, DFSAgent, GreedyAgent, AStarAgent
│       └── visualizer.py           # SearchVisualizer (Dracula theme, 1-by-1 tile sliding, comparison charts)
└── tests/
    └── test_search.py              # Unit tests for BFS, DFS, Greedy, A*, heuristics, and parity
```

---

## 8. Local Setup & Execution Guide

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
* **Launch Interactive Web App (Streamlit):**
  ```bash
  uv run streamlit run app.py
  ```
  Open your browser at `http://localhost:8501`.

* **Launch Native Desktop GUI (Tkinter Canvas Animator):**
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
* **Streamlit Web App:**
  ```powershell
  uv run streamlit run app.py
  ```
* **Native Desktop Canvas GUI:**
  ```powershell
  uv run python gui_app.py
  ```

---

## 9. Streamlit Community Cloud Deployment

To host this project online on [Streamlit Community Cloud](https://share.streamlit.io):

1. Commit and push your code to GitHub:
   ```bash
   git add .
   git commit -m "feat: add Dracula theme, 8-puzzle tile animations, and playback controls"
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **Create app** > **Deploy a public app from GitHub**.
4. Set Repository: `QZainSami/mazemind-ai`.
5. Set Main file path: `app.py`.
6. Click **Deploy!** — Streamlit Cloud automatically detects `requirements.txt` and deploys your application within seconds.
