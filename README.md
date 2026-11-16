# MazeMind AI — Intelligent Puzzle Solver, Evolutionary Level Generator & Search Performance Predictor

[![Course: CSL-325](https://img.shields.io/badge/Course-CSL--325%20AI%20Lab-blue.svg)](https://github.com/QZainSami/mazemind-ai)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-green.svg)](https://python.org)
[![Package Manager: uv](https://img.shields.io/badge/Environment-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Platform: Linux | Windows | macOS](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-orange.svg)](https://github.com/QZainSami/mazemind-ai)
[![Streamlit Cloud Ready](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-red.svg)](https://share.streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Summary & Hard Constraints Adherence

**MazeMind AI** is a unified artificial intelligence laboratory platform integrating classical state-space search, procedural level evolution via genetic algorithms, exploratory data science, and supervised machine learning inference into an interactive multi-tab application.

### Strict Academic Constraints:
* **Zero Pre-Trained Models / External APIs:** All search solvers (BFS, DFS, Greedy, A*), evolutionary operators (tournament selection, single-point crossover, bit-flip mutation), simulation pipelines, and supervised ML models (Decision Tree & Linear Regression) are implemented and trained from scratch in pure Python without black-box APIs.
* **100% Cloud-Deployable:** No external hardware dependencies; ready for one-click deployment on **Streamlit Community Cloud** (`share.streamlit.io`).
* **Cross-Platform Compatibility:** Runs natively across **Linux** (Ubuntu, Debian, Fedora, Arch, WSL), **Windows**, and **macOS**.
* **Deterministic `uv` Packaging:** Dependency management and isolated virtual environments strictly enforced via `uv` (`pyproject.toml` and `uv.lock`), with a frozen `requirements.txt` exported for cloud runtime compatibility.

---

## 2. Theoretical Foundations & Mathematical Formulations

### A. Classical Heuristic Search
1. **Breadth-First Search (BFS):** Unweighted shortest-path guarantee via FIFO queue:
   $$g(n) = \text{depth}(n)$$
2. **Depth-First Search (DFS):** Memory-bounded graph exploration using an explicit LIFO stack with cycle prevention.
3. **Greedy Best-First Search:** Heuristic-driven search prioritizing nodes with minimal estimated distance to goal:
   $$f(n) = h(n)$$
4. **A\* Search:** Optimal heuristic search balancing path cost and admissible future estimates:
   $$f(n) = g(n) + h(n)$$
   * **Admissibility & Consistency:** Using the Manhattan distance metric $h(n) = |r_n - r_{\text{goal}}| + |c_n - c_{\text{goal}}|$ guarantees $h(n) \le h^*(n)$ on orthogonal grid graphs.

### B. 8-Puzzle Solvability & Inversion Parity
For a $3 \times 3$ grid (odd width $W = 3$), blank tile swaps preserve the parity of the non-blank tile permutation. A state is solvable if and only if its inversion count parity matches that of the goal state:

$$\mathrm{Inv}(S) = \sum_{1 \le i < j \le 8} \mathbb{I}[S_i > S_j \land S_i \ne 0 \land S_j \ne 0]$$

$$\mathrm{Solvable}(S, G) \iff \mathrm{Inv}(S) \equiv \mathrm{Inv}(G) \pmod 2$$

### C. Procedural Evolutionary Optimization
The **LevelEvolver** optimizes binary chromosomes of length $R \times C$ to match a target difficulty:
* **Chromosome:** $C \in \{0, 1\}^{R \cdot C}$, with $C_0 = 0$ (Start) and $C_{R \cdot C - 1} = 0$ (Goal).
* **Solvability Penalty:** If unsolvable by A*, assign heavy gradient penalty based on minimum Manhattan distance to goal:
  $$\mathrm{Fitness}_{\mathrm{unsolvable}} = -1000 - 10 \cdot \min_{v \in \mathrm{visited}} d_{\mathrm{Manhattan}}(v, \mathrm{Goal})$$
* **Fitness of Solvable Mazes:**
  $$\mathrm{Fitness} = 100 - 2 \cdot |L_{\mathrm{actual}} - L_{\mathrm{target}}| + 0.5 \cdot (\text{dead ends})$$
* **Operators:** Tournament selection ($k = 3$), single-point crossover ($p_c = 0.8$), bit-flip mutation ($p_m = 0.05$), and elitism ($e = 2$).

### D. Supervised Learning Pipeline
* **Model 1: Decision Tree Classifier:** Predicts `recommended_algo` (`A*`, `Greedy`, or `BFS`) using information gain entropy criterion:
  $$H(S) = -\sum_{i=1}^C p_i \log_2(p_i), \quad \text{max depth} = 5$$
* **Model 2: Linear Regression:** Predicts continuous `astar_path_cost` directly from geometric and topological features:
  $$\hat{y} = \beta_0 + \sum_{j=1}^m \beta_j X_j$$

---

## 3. Repository Architecture

```text
mazemind-ai/
├── pyproject.toml                  # uv project metadata and dependencies
├── uv.lock                         # Deterministic uv lockfile
├── requirements.txt                # Exported via `uv export` for submission/cloud
├── README.md                       # Setup instructions, architecture overview, and live URL
├── LICENSE                         # MIT License (Sami, 2026)
├── app.py                          # Streamlit web application with live animated solving
├── gui_app.py                      # Standalone native desktop Canvas GUI (60 FPS)
├── data/
│   └── maze_benchmarks.csv         # Generated dataset (1,500 simulated maze runs)
├── models/
│   ├── decision_tree.pkl           # Trained DecisionTreeClassifier artifact
│   ├── linear_regression.pkl       # Trained LinearRegression artifact
│   └── model_metrics.json          # Saved evaluation scores (Accuracy, R2, MSE, RMSE)
├── scripts/
│   ├── generate_dataset.py         # Automated simulation script to populate maze_benchmarks.csv
│   └── train_models.py             # Trains & evaluates Decision Tree + Linear Regression models
├── src/
│   └── mazemind/
│       ├── __init__.py
│       ├── environments.py         # OOP classes: Environment, GridMaze, EightPuzzle
│       ├── search_agents.py        # OOP classes: SearchAgent, BFSAgent, DFSAgent, GreedyAgent, AStarAgent
│       ├── genetic_generator.py    # OOP class: LevelEvolver (Genetic Algorithm maze generator)
│       ├── ml_predictor.py         # OOP class: MLPredictor (loads .pkl models for live inference)
│       └── visualizer.py           # Matplotlib figure builders (Headless Agg backend)
└── tests/
    ├── test_search.py              # Unit tests verifying optimality of A*/BFS and 8-puzzle solvability
    └── test_ml.py                  # Unit tests verifying dataset schema and model inference shapes
```

---

## 4. Local Setup & Execution Guide

### 🐧 Linux (Ubuntu, Debian, Fedora, Arch, WSL)

#### Step 1: Install System Dependencies
Install Python 3.12+ and Tkinter (required for the desktop GUI):

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

#### Step 2: Install `uv` Package Manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # or restart your terminal
```

#### Step 3: Clone & Sync
```bash
git clone https://github.com/QZainSami/mazemind-ai.git
cd mazemind-ai

# Synchronize virtual environment deterministically
uv sync
```

#### Step 4: Run Test Suite
```bash
uv run pytest
```

#### Step 5: Launch the Applications
* **Option A — Interactive Streamlit Web App (Recommended for Browser & Cloud):**
  ```bash
  uv run streamlit run app.py
  ```
  Open your browser at `http://localhost:8501`.

* **Option B — Native Desktop Canvas GUI (60 FPS Smooth Animator):**
  ```bash
  uv run python gui_app.py
  ```
  *(Requires an active desktop window session. If using WSL2, ensure WSLg or an X-server like VcXsrv is running).*

---

### 🪟 Windows (PowerShell / Command Prompt)

#### Step 1: Install `uv`
In PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Alternatively via winget:
# winget install astral-sh.uv
```

#### Step 2: Clone & Sync
```powershell
git clone https://github.com/QZainSami/mazemind-ai.git
cd mazemind-ai
uv sync
```

#### Step 3: Run Tests
```powershell
uv run pytest
```

#### Step 4: Launch Applications
* **Streamlit Web App:**
  ```powershell
  uv run streamlit run app.py
  ```
* **Native Desktop Canvas GUI:**
  ```powershell
  uv run python gui_app.py
  ```

---

## 5. Streamlit Community Cloud Deployment

To deploy this project online with zero configuration:

1. Push your repository to GitHub:
   ```bash
   git add .
   git commit -m "feat: complete MazeMind AI architecture"
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and select `<user>/mazemind-ai`.
4. Set **Main file path** to: `app.py`.
5. Click **Deploy!** — Streamlit Cloud detects `requirements.txt` and starts your app in seconds.

---

## 6. Evaluation Metrics Summary

From `models/model_metrics.json`:
* **Dataset Size:** 1,500 solvable mazes (1,200 train / 300 test)
* **Decision Tree Accuracy:** `69.33%` (`criterion="entropy"`, `max_depth=5`)
* **Linear Regression $R^2$ Score:** `0.9993`
* **Linear Regression RMSE:** `0.1680` steps
