"""
Object-Oriented Environments for MazeMind AI.
Includes the Abstract Base Class `Environment`, `GridMaze`, and `EightPuzzle`.
"""

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Environment(ABC):
    """
    Abstract Base Class for search problem environments.
    Enforces a standard interface for state transitions, heuristics, and feature extraction.
    """

    @abstractmethod
    def get_start_state(self) -> Any:
        """Returns the initial state of the environment."""
        pass

    @abstractmethod
    def is_goal_state(self, state: Any) -> bool:
        """Determines if a given state satisfies the goal condition."""
        pass

    @abstractmethod
    def get_neighbors(self, state: Any) -> List[Tuple[Any, int]]:
        """
        Returns a list of tuples (neighbor_state, step_cost)
        reachable from the given state.
        """
        pass

    @abstractmethod
    def heuristic(self, state: Any, metric: str = "manhattan") -> float:
        """Computes heuristic estimate h(n) from state to goal."""
        pass

    @abstractmethod
    def extract_features(self) -> Dict[str, Any]:
        """Extracts numerical features describing the environment for ML models."""
        pass


class GridMaze(Environment):
    """
    2D Grid Maze Environment.
    Grid representation:
        0 = Open passage
        1 = Wall / Obstacle
    States are represented as (row, col) tuples.
    """

    # 4-directional moves: Up, Down, Left, Right
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(
        self,
        grid: np.ndarray,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize GridMaze with a 2D integer NumPy array.
        """
        self.grid = np.array(grid, dtype=np.int32)
        self.rows, self.cols = self.grid.shape
        self.start: Tuple[int, int] = start if start is not None else (0, 0)
        self.goal: Tuple[int, int] = (
            goal if goal is not None else (self.rows - 1, self.cols - 1)
        )

        # Ensure start and goal are open
        self.grid[self.start[0], self.start[1]] = 0
        self.grid[self.goal[0], self.goal[1]] = 0

    def get_start_state(self) -> Tuple[int, int]:
        return self.start

    def is_goal_state(self, state: Tuple[int, int]) -> bool:
        return state == self.goal

    def is_valid_cell(self, r: int, c: int) -> bool:
        """Checks if cell is inside boundary and not a wall."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0

    def get_neighbors(self, state: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """Returns reachable open adjacent cells with unit cost 1."""
        r, c = state
        neighbors: List[Tuple[Tuple[int, int], int]] = []
        for dr, dc in self.DIRECTIONS:
            nr, nc = r + dr, c + dc
            if self.is_valid_cell(nr, nc):
                neighbors.append(((nr, nc), 1))
        return neighbors

    def heuristic(self, state: Tuple[int, int], metric: str = "manhattan") -> float:
        """
        Computes distance heuristic from state to goal.
        Supports 'manhattan' (default) and 'euclidean'.
        """
        r, c = state
        gr, gc = self.goal
        dr = abs(r - gr)
        dc = abs(c - gc)
        if metric == "euclidean":
            return math.sqrt(dr * dr + dc * dc)
        return float(dr + dc)

    def extract_features(self) -> Dict[str, Any]:
        """
        Extracts geometric and topological features for ML inference.
        Returns:
            rows, cols, total_cells, wall_density, manhattan_dist,
            euclidean_dist, dead_end_count, avg_open_neighbors
        """
        total_cells = self.rows * self.cols
        wall_cells = int(np.sum(self.grid == 1))
        wall_density = float(wall_cells / total_cells) if total_cells > 0 else 0.0

        sr, sc = self.start
        gr, gc = self.goal
        manhattan_dist = float(abs(sr - gr) + abs(sc - gc))
        euclidean_dist = float(math.sqrt((sr - gr) ** 2 + (sc - gc) ** 2))

        # Dead ends: open cells having strictly 1 open cardinal neighbor
        # Average open neighbors across all open cells
        open_cell_coords = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r, c] == 0
        ]

        dead_end_count = 0
        open_neighbor_counts: List[int] = []

        for r, c in open_cell_coords:
            open_n = 0
            for dr, dc in self.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if self.is_valid_cell(nr, nc):
                    open_n += 1
            open_neighbor_counts.append(open_n)
            # A dead end is a passage terminal with only 1 exit
            if open_n == 1:
                dead_end_count += 1

        avg_open_neighbors = (
            float(np.mean(open_neighbor_counts)) if open_neighbor_counts else 0.0
        )

        return {
            "rows": int(self.rows),
            "cols": int(self.cols),
            "total_cells": int(total_cells),
            "wall_density": round(wall_density, 4),
            "manhattan_dist": round(manhattan_dist, 4),
            "euclidean_dist": round(euclidean_dist, 4),
            "dead_end_count": int(dead_end_count),
            "avg_open_neighbors": round(avg_open_neighbors, 4),
        }

    def clone(self) -> "GridMaze":
        """Creates a deep copy of the GridMaze."""
        return GridMaze(
            grid=self.grid.copy(),
            start=self.start,
            goal=self.goal,
        )


class EightPuzzle(Environment):
    """
    Classic 8-Puzzle (3x3 Sliding Tile Puzzle) Environment.
    State representation: Immutable tuple of length 9 containing numbers 0..8,
    where 0 denotes the blank tile.
    Example: (1, 2, 3, 4, 5, 6, 7, 8, 0)
    """

    STANDARD_GOAL: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    SPIRAL_GOAL: Tuple[int, ...] = (1, 2, 3, 8, 0, 4, 7, 6, 5)

    def __init__(
        self,
        start_state: Tuple[int, ...],
        goal_state: Optional[Tuple[int, ...]] = None,
    ):
        if len(start_state) != 9:
            raise ValueError("Start state must be a tuple of 9 integers.")
        self.start_state = tuple(start_state)
        self.goal_state = (
            tuple(goal_state) if goal_state is not None else self.STANDARD_GOAL
        )

    def get_start_state(self) -> Tuple[int, ...]:
        return self.start_state

    def is_goal_state(self, state: Tuple[int, ...]) -> bool:
        return state == self.goal_state

    @staticmethod
    def count_inversions(state: Tuple[int, ...]) -> int:
        """
        Counts inversions in the state, ignoring the blank tile (0).
        An inversion occurs when a tile appears before another tile with a smaller number.
        """
        tiles = [t for t in state if t != 0]
        inversions = 0
        n = len(tiles)
        for i in range(n):
            for j in range(i + 1, n):
                if tiles[i] > tiles[j]:
                    inversions += 1
        return inversions

    @classmethod
    def is_solvable(
        cls,
        state: Tuple[int, ...],
        goal_state: Optional[Tuple[int, ...]] = None,
    ) -> bool:
        """
        Checks solvability using inversion parity.
        For odd grid width (3x3), a state is solvable if and only if
        its inversion count parity matches the goal state parity.
        """
        target = goal_state if goal_state is not None else cls.STANDARD_GOAL
        return (cls.count_inversions(state) % 2) == (cls.count_inversions(target) % 2)

    def get_neighbors(self, state: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], int]]:
        """
        Generates valid successor states by sliding tiles into the blank position (0).
        Returns list of (neighbor_state, step_cost=1).
        """
        blank_idx = state.index(0)
        r, c = divmod(blank_idx, 3)

        neighbors: List[Tuple[Tuple[int, ...], int]] = []
        # Possible blank moves: Up (-1, 0), Down (1, 0), Left (0, -1), Right (0, 1)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                target_idx = nr * 3 + nc
                new_state = list(state)
                # Swap blank with tile
                new_state[blank_idx], new_state[target_idx] = (
                    new_state[target_idx],
                    new_state[blank_idx],
                )
                neighbors.append((tuple(new_state), 1))

        return neighbors

    def heuristic(self, state: Tuple[int, ...], metric: str = "manhattan") -> float:
        """
        Computes heuristic distance from state to goal.
        - 'manhattan': Sum of Manhattan distances of all numbered tiles (excluding 0)
          to their positions in goal_state. Admissible & consistent.
        - 'misplaced': Count of non-zero tiles not in their goal position.
        """
        if metric == "misplaced":
            return float(
                sum(
                    1
                    for i in range(9)
                    if state[i] != 0 and state[i] != self.goal_state[i]
                )
            )

        # Manhattan distance
        total_dist = 0
        goal_positions = {val: divmod(idx, 3) for idx, val in enumerate(self.goal_state)}

        for idx, val in enumerate(state):
            if val != 0:
                cur_r, cur_c = divmod(idx, 3)
                goal_r, goal_c = goal_positions[val]
                total_dist += abs(cur_r - goal_r) + abs(cur_c - goal_c)

        return float(total_dist)

    def extract_features(self) -> Dict[str, Any]:
        """Extracts features for the 8-puzzle board."""
        inversions = self.count_inversions(self.start_state)
        solvable = self.is_solvable(self.start_state, self.goal_state)
        manhattan_h = self.heuristic(self.start_state, metric="manhattan")
        misplaced_h = self.heuristic(self.start_state, metric="misplaced")

        return {
            "inversion_count": inversions,
            "is_solvable": solvable,
            "initial_manhattan": manhattan_h,
            "initial_misplaced": misplaced_h,
        }
