# MazeMind AI — Intelligent Puzzle Solver, Evolutionary Level Generator & Search Performance Predictor

[![Course: CSL-325](https://img.shields.io/badge/Course-CSL--325%20AI%20Lab-blue.svg)](https://github.com)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-green.svg)](https://python.org)
[![Package Manager: uv](https://img.shields.io/badge/Environment-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Streamlit Cloud Ready](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-red.svg)](https://streamlit.io)

---

## 1. Executive Summary & Hard Constraints Adherence

**MazeMind AI** is a unified artificial intelligence laboratory platform integrating classical state-space search, procedural level evolution via genetic algorithms, exploratory data science, and supervised machine learning inference into a single interactive Streamlit application.

### Strict Academic Constraints:
* **Zero Pre-Trained Models / External APIs:** All search solvers (BFS, DFS, Greedy, A*), evolutionary operators (tournament selection, single-point crossover, bit-flip mutation), simulation pipelines, and supervised ML models (Decision Tree & Linear Regression) are trained and implemented from scratch in pure Python without black-box APIs.
* **100% Cloud-Deployable:** No external hardware dependencies or native graphical drivers required; ready for one-click deployment on **Streamlit Community Cloud** (`share.streamlit.io`).
* **Deterministic `uv` Packaging:** Dependency management and isolated virtual environments strictly enforced via `uv` (`pyproject.toml` and `uv.lock`), with a frozen `requirements.txt` exported for cloud runtime compatibility.

---

## 2. Theoretical Foundations & Mathematical Formulations

### A. Classical Heuristic Search
1. **Breadth-First Search (BFS):** Unweighted shortest-path guarantee via FIFO queue:
   $$\text{Path Cost } g(n) = \text{depth}(n)$$
2. **Depth-First Search (DFS):** Memory-bounded graph exploration using an explicit LIFO stack with strict cycle prevention.
3. **Greedy Best-First Search:** Heuristic-driven search prioritizing nodes with minimal estimated distance to goal:
   $$f(n) = h(n)$$
4. **A\* Search:** Optimal heuristic search balancing path cost and admissible future estimates:
   $$f(n) = g(n) + h(n)$$
   - **Admissibility & Consistency:** Using the Manhattan distance metric $h(n) = |r_n - r_{\text{goal}}| + |c_n - c_{\text{goal}}|$ guarantees $h(n) \le h^*(n)$ on grid topologies with orthogonal unit transitions.

### B. 8-Puzzle Solvability & Inversion Parity
For a 3×3 grid (odd width $W = 3$), the blank tile swaps do not alter the parity of the non-blank tile permutation. A state is solvable if and only if its inversion count parity matches that of the goal state:

$$\text{Inv}(S) = \sum_{1 \le i < j \le 8} \mathbb{I}[S_i > S_j \land S_i \ne 0 \land S_j \ne 0]$$

$$\text{Solvable}(S, G) \iff \text{Inv}(S) \equiv \text{Inv}(G) \pmod 2$$

### C. Procedural Evolutionary Optimization
The **LevelEvolver** optimizes binary chromosomes of length $R \times C$ to match a target difficulty:
* **Chromosome:** $C \in \{0, 1\}^{R \cdot C}$, with $C_0 = 0$ (Start) and $C_{R \cdot C - 1} = 0$ (Goal).
* **Solvability Penalty:** If unsolvable by A*, assign heavy gradient penalty based on minimum Manhattan distance to goal:
  $$\text{Fitness}_{\text{unsolvable}} = -1000 - 10 \cdot \min_{v \in \text{visited}} d_{\text{Manhattan}}(v, \text{Goal})$$
* **Fitness of Solvable Mazes:**
  $$\text{Fitness} = 100 - 2 \cdot |L_{\text{actual}} - L_{\text{target}}| + 0.5 \cdot \text{dead\_ends}$$
* **Operators:** Tournament selection ($k = 3$), single-point crossover ($p_c = 0.8$), bit-flip mutation ($p_m = 0.05$), and elitism ($e = 2$).

### D. Supervised Learning Pipeline
* **Model 1: Decision Tree Classifier:** Predicts `recommended_algo` (`A*`, `Greedy`, or `BFS`) using information gain entropy criterion:
  $$H(S) = -\sum_{i=1}^C p_i \log_2(p_i), \quad \text{max\_depth} = 5$$
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
├── app.py                          # Main Streamlit web application entrypoint
├── data/
│   └── maze_benchmarks.csv         # Generated dataset (1,500+ simulated maze runs)
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
│       └── visualizer.py           # Matplotlib figure builders for grids, trees, and evaluation plots
└── tests/
    ├── test_search.py              # Unit tests verifying optimality of A*/BFS and 8-puzzle solvability
    └── test_ml.py                  # Unit tests verifying dataset schema and model inference shapes
```

---

## 4. Local Setup & Execution Guide

### Prerequisites
* Python 3.12+
* `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `winget install astral-sh.uv`)

### 1. Clone & Sync Environment
```bash
git clone <your-repo-url>
cd "MazeMind AI"

# Install all dependencies exactly as locked
uv sync
```

### 2. Run Test Suite
Verify algorithmic optimality and dataset integrity:
```bash
uv run pytest
```

### 3. (Optional) Re-Generate Dataset & Re-Train Models
```bash
# Generate 1,500 solvable benchmark mazes
uv run python scripts/generate_dataset.py

# Train Decision Tree and Linear Regression models
uv run python scripts/train_models.py
```

### 4. Launch Applications

#### A. Interactive Streamlit Web Application (Cloud & Local)
Features all 4 academic tabs, automated live solving animations with speed controls, GA level generation, dataset query explorer, and ML training:
```bash
uv run streamlit run app.py
```
Open your browser at `http://localhost:8501`.

#### B. Native Desktop Canvas GUI (Ultra-Smooth 60 FPS Desktop App)
Features mouse drag wall drawing, real-time animated search with 1ms–150ms speed control, live ML pre-scan, and pause/step controls:
```bash
uv run python gui_app.py
```

---

## 5. Streamlit Community Cloud Deployment

To deploy on [Streamlit Community Cloud](https://share.streamlit.io):

1. Push your repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "feat: complete MazeMind AI architecture"
   git remote add origin https://github.com/<username>/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repository.
4. Set Main file path: `app.py`.
5. Streamlit Cloud will automatically detect `requirements.txt` and install all necessary dependencies.

---

## 6. Evaluation Metrics Summary

From `models/model_metrics.json`:
* **Dataset Size:** 1,500 solvable mazes (1,200 train / 300 test)
* **Decision Tree Accuracy:** `69.33%` (`criterion="entropy"`, `max_depth=5`)
* **Linear Regression $R^2$ Score:** `0.9993`
* **Linear Regression RMSE:** `0.1680` steps
