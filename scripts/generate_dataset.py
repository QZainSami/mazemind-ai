"""
Dataset Generation Script for MazeMind AI.
Generates 1,500+ diverse, solvable mazes with varied dimensions and obstacle densities.
Runs BFS, DFS, Greedy, and A* search agents to extract features and ground-truth targets.
Saves records to `data/maze_benchmarks.csv`.
"""

import os
import random
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mazemind.environments import GridMaze
from mazemind.search_agents import BFSAgent, DFSAgent, GreedyAgent, AStarAgent


def create_solvable_maze(
    rows: int,
    cols: int,
    wall_density: float,
    rng: np.random.Generator,
) -> GridMaze:
    """
    Creates a guaranteed solvable maze by carving a random path from start to goal
    and distributing walls according to the requested density.
    """
    # 0 = open, 1 = wall
    # Start with empty grid
    grid = np.zeros((rows, cols), dtype=np.int32)

    # 1. Carve a randomized path from (0, 0) to (rows - 1, cols - 1)
    curr_r, curr_c = 0, 0
    guaranteed_path = {(0, 0)}

    while (curr_r, curr_c) != (rows - 1, cols - 1):
        moves = []
        if curr_r < rows - 1:
            moves.append((curr_r + 1, curr_c))
        if curr_c < cols - 1:
            moves.append((curr_r, curr_c + 1))
        # Occasional random wander to increase maze complexity
        if rng.random() < 0.25:
            if curr_r > 0:
                moves.append((curr_r - 1, curr_c))
            if curr_c > 0:
                moves.append((curr_r, curr_c - 1))

        next_r, next_c = moves[rng.integers(0, len(moves))]
        curr_r, curr_c = next_r, next_c
        guaranteed_path.add((curr_r, curr_c))

    # 2. Populate remaining cells with walls to match wall_density
    total_cells = rows * cols
    target_walls = int(total_cells * wall_density)

    eligible_cells = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in guaranteed_path
    ]

    if eligible_cells and target_walls > 0:
        num_walls = min(target_walls, len(eligible_cells))
        wall_indices = rng.choice(len(eligible_cells), size=num_walls, replace=False)
        for idx in wall_indices:
            wr, wc = eligible_cells[idx]
            grid[wr, wc] = 1

    return GridMaze(grid=grid, start=(0, 0), goal=(rows - 1, cols - 1))


def determine_recommended_algo(
    wall_density: float,
    astar_cost: int,
    astar_nodes: int,
    greedy_cost: int,
    greedy_nodes: int,
    bfs_nodes: int,
) -> str:
    """
    Selects recommended algorithm based on benchmark heuristics:
    - 'BFS': on ultra-low-density grids (density < 0.16) where unweighted expansion is fast
    - 'Greedy': when Greedy finds near-optimal path (cost <= 1.08x A*) with >= 30% fewer nodes
    - 'A*': when optimality and guided low expansion wins
    """
    if wall_density < 0.16:
        return "BFS"

    if greedy_cost <= int(astar_cost * 1.08) and greedy_nodes <= int(astar_nodes * 0.70):
        return "Greedy"

    return "A*"


def generate_benchmark_dataset(
    num_samples: int = 1500,
    output_path: str = "data/maze_benchmarks.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """Generates the benchmark dataset and writes to CSV."""
    print(f"Generating {num_samples} solvable maze benchmarks (seed={seed})...")
    rng = np.random.default_rng(seed)

    bfs_agent = BFSAgent()
    dfs_agent = DFSAgent()
    greedy_agent = GreedyAgent(heuristic_metric="manhattan")
    astar_agent = AStarAgent(heuristic_metric="manhattan")

    records: List[Dict] = []

    for i in range(1, num_samples + 1):
        # Randomized dimensions (10x10 to 25x25)
        rows = int(rng.integers(10, 26))
        cols = int(rng.integers(10, 26))
        # Randomized obstacle density (0.10 to 0.38)
        wall_density = round(float(rng.uniform(0.10, 0.38)), 4)

        maze = create_solvable_maze(rows, cols, wall_density, rng)
        features = maze.extract_features()

        # Run solvers
        bfs_res = bfs_agent.solve(maze)
        dfs_res = dfs_agent.solve(maze)
        greedy_res = greedy_agent.solve(maze)
        astar_res = astar_agent.solve(maze)

        # Ensure maze is solvable
        assert astar_res.found, f"Maze {i} was generated unsolvable!"

        recommended_algo = determine_recommended_algo(
            wall_density=features["wall_density"],
            astar_cost=astar_res.path_cost,
            astar_nodes=astar_res.nodes_expanded,
            greedy_cost=greedy_res.path_cost,
            greedy_nodes=greedy_res.nodes_expanded,
            bfs_nodes=bfs_res.nodes_expanded,
        )

        record = {
            "maze_id": i,
            # Geometric & Topological Features
            "rows": features["rows"],
            "cols": features["cols"],
            "total_cells": features["total_cells"],
            "wall_density": features["wall_density"],
            "manhattan_dist": features["manhattan_dist"],
            "euclidean_dist": features["euclidean_dist"],
            "dead_end_count": features["dead_end_count"],
            "avg_open_neighbors": features["avg_open_neighbors"],
            # Solver Performance Metrics
            "astar_path_cost": astar_res.path_cost,
            "astar_nodes_expanded": astar_res.nodes_expanded,
            "astar_time_ms": astar_res.execution_time_ms,
            "bfs_path_cost": bfs_res.path_cost,
            "bfs_nodes_expanded": bfs_res.nodes_expanded,
            "bfs_time_ms": bfs_res.execution_time_ms,
            "dfs_path_cost": dfs_res.path_cost,
            "dfs_nodes_expanded": dfs_res.nodes_expanded,
            "dfs_time_ms": dfs_res.execution_time_ms,
            "greedy_path_cost": greedy_res.path_cost,
            "greedy_nodes_expanded": greedy_res.nodes_expanded,
            "greedy_time_ms": greedy_res.execution_time_ms,
            # Ground-truth ML Target
            "recommended_algo": recommended_algo,
        }
        records.append(record)

        if i % 250 == 0 or i == num_samples:
            print(f"  Progress: {i}/{num_samples} mazes simulated.")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to '{output_path}'. Total rows: {len(df)}")
    print("Class distribution for 'recommended_algo':")
    print(df["recommended_algo"].value_counts())
    return df


if __name__ == "__main__":
    generate_benchmark_dataset()
