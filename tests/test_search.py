"""
Unit Tests for Search Algorithms and Environments.
Verifies optimality of A* and BFS, DFS and Greedy behavior,
and 8-Puzzle solvability and heuristic correctness.
"""

import pytest
import numpy as np

from mazemind.environments import GridMaze, EightPuzzle
from mazemind.search_agents import BFSAgent, DFSAgent, GreedyAgent, AStarAgent


def test_grid_maze_empty_optimality():
    """Verify BFS and A* find the exact same optimal path length on an open grid."""
    grid = np.zeros((8, 8), dtype=np.int32)
    maze = GridMaze(grid=grid, start=(0, 0), goal=(7, 7))

    bfs = BFSAgent()
    astar = AStarAgent()

    bfs_res = bfs.solve(maze)
    astar_res = astar.solve(maze)

    assert bfs_res.found is True
    assert astar_res.found is True
    # Manhattan distance on 8x8 is 14 steps
    assert bfs_res.path_cost == 14
    assert astar_res.path_cost == 14
    assert len(bfs_res.path) == len(astar_res.path)


def test_grid_maze_with_obstacle_wall():
    """Verify BFS and A* find equal optimal paths around an obstacle wall."""
    grid = np.zeros((7, 7), dtype=np.int32)
    # Vertical wall down the middle leaving a gap at bottom
    grid[0:5, 3] = 1

    maze = GridMaze(grid=grid, start=(0, 0), goal=(0, 6))

    bfs = BFSAgent()
    astar = AStarAgent()
    dfs = DFSAgent()
    greedy = GreedyAgent()

    bfs_res = bfs.solve(maze)
    astar_res = astar.solve(maze)
    dfs_res = dfs.solve(maze)
    greedy_res = greedy.solve(maze)

    assert bfs_res.found is True
    assert astar_res.found is True
    assert dfs_res.found is True
    assert greedy_res.found is True

    # A* and BFS must yield identical optimal path length
    assert astar_res.path_cost == bfs_res.path_cost
    # Path should start at start and end at goal
    assert astar_res.path[0] == (0, 0)
    assert astar_res.path[-1] == (0, 6)


def test_unsolvable_grid_maze():
    """Verify all agents handle an enclosed, unsolvable goal gracefully."""
    grid = np.zeros((5, 5), dtype=np.int32)
    # Completely box in the goal (4, 4)
    grid[3, 4] = 1
    grid[4, 3] = 1

    maze = GridMaze(grid=grid, start=(0, 0), goal=(4, 4))

    for agent in [BFSAgent(), DFSAgent(), GreedyAgent(), AStarAgent()]:
        res = agent.solve(maze)
        assert res.found is False
        assert res.path == []
        assert res.path_cost == 0


def test_eight_puzzle_solvability():
    """Verify 8-puzzle inversion parity detection."""
    # Standard goal: (1, 2, 3, 4, 5, 6, 7, 8, 0) -> 0 inversions (even -> solvable)
    solvable_state = (1, 2, 3, 4, 5, 6, 0, 7, 8)
    assert EightPuzzle.is_solvable(solvable_state) is True

    # Unsolvable state: swap 1 and 2 in goal state -> 1 inversion (odd -> unsolvable)
    unsolvable_state = (2, 1, 3, 4, 5, 6, 7, 8, 0)
    assert EightPuzzle.is_solvable(unsolvable_state) is False


def test_eight_puzzle_astar_solution():
    """Verify A* solves a known 8-puzzle board with minimal steps."""
    # Move blank right from (1, 2, 3, 4, 5, 6, 7, 0, 8) to goal
    start = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    puzzle = EightPuzzle(start_state=start)

    astar = AStarAgent()
    res = astar.solve(puzzle)

    assert res.found is True
    assert res.path_cost == 1
    assert res.path[-1] == EightPuzzle.STANDARD_GOAL


def test_eight_puzzle_multistep_solution():
    """Verify A* and BFS find optimal path for a 3-step puzzle."""
    # Standard goal: (1, 2, 3, 4, 5, 6, 7, 8, 0)
    # Start: blank moved left, left, up
    # (1, 2, 3, 4, 0, 5, 7, 8, 6)
    start = (1, 2, 3, 4, 0, 6, 7, 5, 8)
    puzzle = EightPuzzle(start_state=start)

    assert puzzle.is_solvable(start) is True

    astar = AStarAgent(heuristic_metric="manhattan")
    bfs = BFSAgent()

    astar_res = astar.solve(puzzle)
    bfs_res = bfs.solve(puzzle)

    assert astar_res.found is True
    assert bfs_res.found is True
    assert astar_res.path_cost == bfs_res.path_cost
