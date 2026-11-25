"""
Unit Tests for Search Algorithms and Environments.
Verifies:
- Optimality of BFS and A* (with Manhattan and Euclidean heuristics)
- DFS path finding and cycle prevention
- Greedy Best-First Search heuristics
- 8-Puzzle solvability inversion parity math
- 8-Puzzle A* and BFS solving
Course: CSL-325 Artificial Intelligence Lab (Fall 2026).
"""

import pytest
import numpy as np

from mazemind.environments import GridMaze, EightPuzzle
from mazemind.search_agents import BFSAgent, DFSAgent, GreedyAgent, AStarAgent
from mazemind.web_animator import build_maze_animator_html, build_puzzle_animator_html


def test_grid_maze_empty_optimality():
    """Verify BFS and A* find the exact same optimal path length on an open grid."""
    grid = np.zeros((8, 8), dtype=np.int32)
    maze = GridMaze(grid=grid, start=(0, 0), goal=(7, 7))

    bfs = BFSAgent()
    astar_manhattan = AStarAgent(heuristic_metric="manhattan")
    astar_euclidean = AStarAgent(heuristic_metric="euclidean")

    bfs_res = bfs.solve(maze)
    astar_m_res = astar_manhattan.solve(maze)
    astar_e_res = astar_euclidean.solve(maze)

    assert bfs_res.found is True
    assert astar_m_res.found is True
    assert astar_e_res.found is True
    # Manhattan distance on 8x8 is 14 steps
    assert bfs_res.path_cost == 14
    assert astar_m_res.path_cost == 14
    assert astar_e_res.path_cost == 14
    assert len(bfs_res.path) == len(astar_m_res.path)


def test_grid_maze_with_obstacle_wall():
    """Verify BFS and A* find equal optimal paths around an obstacle wall."""
    grid = np.zeros((7, 7), dtype=np.int32)
    # Vertical wall down the middle leaving a gap at bottom
    grid[0:5, 3] = 1

    maze = GridMaze(grid=grid, start=(0, 0), goal=(0, 6))

    bfs = BFSAgent()
    astar = AStarAgent(heuristic_metric="manhattan")
    dfs = DFSAgent()
    greedy = GreedyAgent(heuristic_metric="manhattan")

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
    assert astar_res.path[0] == (0, 0)
    assert astar_res.path[-1] == (0, 6)


def test_unsolvable_grid_maze():
    """Verify all agents handle an enclosed, unsolvable goal gracefully."""
    grid = np.zeros((5, 5), dtype=np.int32)
    # Completely box in the goal (4, 4)
    grid[3, 4] = 1
    grid[4, 3] = 1

    maze = GridMaze(grid=grid, start=(0, 0), goal=(4, 4))

    for agent in [
        BFSAgent(),
        DFSAgent(),
        GreedyAgent(heuristic_metric="manhattan"),
        AStarAgent(heuristic_metric="manhattan"),
    ]:
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
    start = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    puzzle = EightPuzzle(start_state=start)

    astar = AStarAgent(heuristic_metric="manhattan")
    res = astar.solve(puzzle)

    assert res.found is True
    assert res.path_cost == 1
    assert res.path[-1] == EightPuzzle.STANDARD_GOAL


def test_eight_puzzle_multistep_solution():
    """Verify A* and BFS find optimal path for a 3-step puzzle."""
    start = (1, 2, 3, 4, 0, 6, 7, 5, 8)
    puzzle = EightPuzzle(start_state=start)

    assert puzzle.is_solvable(start) is True

    astar_m = AStarAgent(heuristic_metric="manhattan")
    astar_e = AStarAgent(heuristic_metric="euclidean")
    bfs = BFSAgent()

    astar_m_res = astar_m.solve(puzzle)
    astar_e_res = astar_e.solve(puzzle)
    bfs_res = bfs.solve(puzzle)

    assert astar_m_res.found is True
    assert astar_e_res.found is True
    assert bfs_res.found is True
    assert astar_m_res.path_cost == bfs_res.path_cost
    assert astar_e_res.path_cost == bfs_res.path_cost


def test_maze_and_puzzle_stats():
    """Verify environment statistics dictionaries."""
    grid = np.zeros((6, 6), dtype=np.int32)
    grid[2, 2] = 1
    maze = GridMaze(grid=grid, start=(0, 0), goal=(5, 5))
    stats = maze.get_stats()

    assert stats["rows"] == 6
    assert stats["cols"] == 6
    assert stats["total_cells"] == 36
    assert stats["wall_cells"] == 1
    assert stats["manhattan_dist"] == 10.0

    puzzle = EightPuzzle(start_state=(1, 2, 3, 4, 0, 6, 7, 5, 8))
    p_stats = puzzle.get_stats()
    assert p_stats["is_solvable"] is True
    assert p_stats["inversion_count"] % 2 == 0


def test_guaranteed_solvable_maze_at_max_size_and_density():
    """Verify generated mazes stay solvable even at max grid size (35x35) and high obstacle density (0.50)."""
    astar = AStarAgent(heuristic_metric="manhattan")
    for preset in ["Random Obstacles", "Open Room (No Walls)", "Center Wall Barrier", "Snake Labyrinth"]:
        for trial in range(5):
            maze = GridMaze.generate_solvable(
                rows=35,
                cols=35,
                density=0.50,
                preset=preset,
                seed=100 + trial,
            )
            res = astar.solve(maze)
            assert res.found is True, f"Preset {preset} trial {trial} produced unsolvable 35x35 maze"
            assert res.path[0] == maze.start
            assert res.path[-1] == maze.goal


def test_web_animator_html_embeds_precomputed_data():
    """Verify web animator HTML renders cleanly with precomputed search results."""
    maze = GridMaze.generate_solvable(rows=10, cols=10, density=0.35, seed=7)
    res = AStarAgent(heuristic_metric="manhattan").solve(maze)
    html_maze = build_maze_animator_html(
        maze=maze,
        results={"A* (Manhattan)": res},
        autoplay=True,
        instant=False,
        run_id=1,
    )
    assert "mazeCanvas" in html_maze
    assert "A* (Manhattan)" in html_maze
    assert '"found": true' in html_maze
    assert '"runId": 1' in html_maze

    puzzle = EightPuzzle(start_state=(1, 2, 3, 4, 0, 6, 7, 5, 8))
    p_res = AStarAgent(heuristic_metric="manhattan").solve(puzzle)
    html_puzzle = build_puzzle_animator_html(
        start_state=puzzle.start_state,
        goal_state=puzzle.goal_state,
        results={"A* (Manhattan)": p_res},
        autoplay=True,
        instant=False,
        run_id=2,
    )
    assert "puzzleCanvas" in html_puzzle
    assert "Tile 5 slid" in html_puzzle
    assert '"runId": 2' in html_puzzle
