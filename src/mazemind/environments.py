"""
Object-Oriented Environments for MazeMind AI.
Includes the Abstract Base Class `Environment`, `GridMaze`, and `EightPuzzle`.
Designed for CSL-325 Artificial Intelligence Lab (Fall 2026).
"""

from abc import ABC, abstractmethod
from collections import deque
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Environment(ABC):
    """
    Abstract Base Class for state-space search environments.
    Enforces a standard object-oriented interface for state transitions,
    heuristics, and problem statistics.
    """

    @abstractmethod
    def get_start_state(self) -> Any:
        """Returns the initial state of the environment."""
        pass

    @abstractmethod
    def get_goal_state(self) -> Any:
        """Returns the target goal state of the environment."""
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

    def get_reverse_neighbors(self, state: Any) -> List[Tuple[Any, int]]:
        """
        Returns predecessor states (prev_state, edge_cost(prev -> state))
        for backward search in bidirectional algorithms.
        """
        return self.get_neighbors(state)

    @abstractmethod
    def heuristic(
        self,
        state: Any,
        metric: str = "manhattan",
        target: Optional[Any] = None,
    ) -> float:
        """Computes heuristic estimate h(n) from state to target (defaults to goal)."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Returns descriptive statistics of the environment."""
        pass


class GridMaze(Environment):
    """
    2D Grid Maze Environment with optional Weighted Terrain costs.
    Grid representation:
        0 = Passable cell (Open cost=1, Mud cost=4, Water cost=8 via `weights`)
        1 = Wall / Obstacle
    States are represented as (row, col) tuples.
    """

    # 4-directional moves: Up, Down, Left, Right (step cost = 1)
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(
        self,
        grid: np.ndarray,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ):
        """Initialize GridMaze with a 2D integer NumPy array (0=open, 1=wall)."""
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

    def get_goal_state(self) -> Tuple[int, int]:
        return self.goal

    def is_goal_state(self, state: Tuple[int, int]) -> bool:
        return state == self.goal

    def is_valid_cell(self, r: int, c: int) -> bool:
        """Checks if cell is inside boundary and not a wall."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0

    def get_neighbors(self, state: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """Returns reachable open adjacent cells with unit step cost 1."""
        r, c = state
        rows, cols, grid = self.rows, self.cols, self.grid
        neighbors: List[Tuple[Tuple[int, int], int]] = []
        if r > 0 and grid[r - 1, c] == 0:
            neighbors.append(((r - 1, c), 1))
        if r + 1 < rows and grid[r + 1, c] == 0:
            neighbors.append(((r + 1, c), 1))
        if c > 0 and grid[r, c - 1] == 0:
            neighbors.append(((r, c - 1), 1))
        if c + 1 < cols and grid[r, c + 1] == 0:
            neighbors.append(((r, c + 1), 1))
        return neighbors

    def heuristic(
        self,
        state: Tuple[int, int],
        metric: str = "manhattan",
        target: Optional[Tuple[int, int]] = None,
    ) -> float:
        """
        Computes admissible distance heuristic from state to target (defaults to goal).
        - 'manhattan': |r - gr| + |c - gc|
        - 'euclidean': sqrt((r - gr)^2 + (c - gc)^2)
        """
        r, c = state
        gr, gc = target if target is not None else self.goal
        dr = abs(r - gr)
        dc = abs(c - gc)
        if metric == "euclidean":
            return float(math.sqrt(dr * dr + dc * dc))
        return float(dr + dc)

    def get_stats(self) -> Dict[str, Any]:
        """Returns vectorized topological and geometric characteristics of the maze."""
        rows, cols = self.rows, self.cols
        total_cells = rows * cols
        open_mask = self.grid == 0
        open_cells = int(np.sum(open_mask))
        wall_cells = total_cells - open_cells
        wall_density = float(wall_cells / total_cells) if total_cells > 0 else 0.0

        sr, sc = self.start
        gr, gc = self.goal
        manhattan_dist = float(abs(sr - gr) + abs(sc - gc))
        euclidean_dist = float(math.sqrt((sr - gr) ** 2 + (sc - gc) ** 2))

        # Vectorized neighbor count for all open cells
        neighbor_counts = np.zeros((rows, cols), dtype=np.int32)
        if rows > 1:
            neighbor_counts[1:, :] += open_mask[:-1, :]
            neighbor_counts[:-1, :] += open_mask[1:, :]
        if cols > 1:
            neighbor_counts[:, 1:] += open_mask[:, :-1]
            neighbor_counts[:, :-1] += open_mask[:, 1:]

        open_nbr_vals = neighbor_counts[open_mask]
        dead_end_count = int(np.sum(open_nbr_vals == 1))
        avg_open_neighbors = (
            float(np.mean(open_nbr_vals)) if open_nbr_vals.size > 0 else 0.0
        )

        return {
            "rows": int(rows),
            "cols": int(cols),
            "total_cells": int(total_cells),
            "open_cells": int(open_cells),
            "wall_cells": int(wall_cells),
            "wall_density": round(wall_density, 4),
            "manhattan_dist": round(manhattan_dist, 4),
            "euclidean_dist": round(euclidean_dist, 4),
            "dead_end_count": int(dead_end_count),
            "avg_open_neighbors": round(avg_open_neighbors, 4),
        }

    def extract_features(self) -> Dict[str, Any]:
        """Backward compatibility alias for get_stats()."""
        return self.get_stats()

    def clone(self) -> "GridMaze":
        """Creates a deep copy of the GridMaze."""
        return GridMaze(
            grid=self.grid.copy(),
            start=self.start,
            goal=self.goal,
        )

    @staticmethod
    def _is_reachable(
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> bool:
        """Checks if goal is reachable from start via open (0) cells."""
        rows, cols = grid.shape
        if grid[start] != 0 or grid[goal] != 0:
            return False
        if start == goal:
            return True

        queue: deque[Tuple[int, int]] = deque([start])
        seen = {start}
        while queue:
            r, c = queue.popleft()
            for dr, dc in GridMaze.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and grid[nr, nc] == 0
                    and (nr, nc) not in seen
                ):
                    if (nr, nc) == goal:
                        return True
                    seen.add((nr, nc))
                    queue.append((nr, nc))
        return False

    @staticmethod
    def _carve_minimum_path_01_bfs(
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Carves a guaranteed path from start to goal by clearing only the
        minimum number of blocking walls via randomized 0-1 BFS.
        Preserves maze density and winding corridor topology even at high densities.
        """
        rows, cols = grid.shape
        carved = grid.copy()
        carved[start] = 0
        carved[goal] = 0

        inf = rows * cols + 1
        dist = np.full((rows, cols), inf, dtype=np.int32)
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        dist[start] = 0
        dq: deque[Tuple[int, int, int]] = deque([(0, start[0], start[1])])
        dirs = list(GridMaze.DIRECTIONS)

        while dq:
            d, r, c = dq.popleft()
            if d > dist[r, c]:
                continue
            if (r, c) == goal:
                break

            rng.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    w = int(carved[nr, nc])
                    nd = d + w
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        parent[(nr, nc)] = (r, c)
                        if w == 0:
                            dq.appendleft((nd, nr, nc))
                        else:
                            dq.append((nd, nr, nc))

        curr: Optional[Tuple[int, int]] = goal
        while curr is not None:
            carved[curr] = 0
            curr = parent.get(curr)

        return carved

    @classmethod
    def _generate_recursive_backtracker(
        cls,
        rows: int,
        cols: int,
        start: Tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generates a classic 1-cell-wide winding labyrinth via iterative DFS."""
        grid = np.ones((rows, cols), dtype=np.int32)
        sr, sc = start
        grid[sr, sc] = 0
        stack: List[Tuple[int, int]] = [(sr, sc)]
        dirs2 = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        while stack:
            r, c = stack[-1]
            candidates = []
            for dr, dc in dirs2:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
                    candidates.append((nr, nc, r + dr // 2, c + dc // 2))
            if candidates:
                idx = int(rng.integers(0, len(candidates)))
                nr, nc, wr, wc = candidates[idx]
                grid[wr, wc] = 0
                grid[nr, nc] = 0
                stack.append((nr, nc))
            else:
                stack.pop()
        return grid

    @classmethod
    def _generate_prims_maze(
        cls,
        rows: int,
        cols: int,
        start: Tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generates a branching labyrinth using Randomized Prim's algorithm."""
        grid = np.ones((rows, cols), dtype=np.int32)
        sr, sc = start
        grid[sr, sc] = 0
        dirs2 = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        frontier: List[Tuple[int, int]] = []
        in_frontier = set()

        def add_frontier(r: int, c: int) -> None:
            for dr, dc in dirs2:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
                    if (nr, nc) not in in_frontier:
                        in_frontier.add((nr, nc))
                        frontier.append((nr, nc))

        add_frontier(sr, sc)
        while frontier:
            pick = int(rng.integers(0, len(frontier)))
            r, c = frontier.pop(pick)
            in_frontier.discard((r, c))

            open_nbrs = []
            for dr, dc in dirs2:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    open_nbrs.append((r + dr // 2, c + dc // 2))

            if open_nbrs and grid[r, c] == 1:
                wr, wc = open_nbrs[int(rng.integers(0, len(open_nbrs)))]
                grid[wr, wc] = 0
                grid[r, c] = 0
                add_frontier(r, c)
        return grid

    @classmethod
    def generate_solvable(
        cls,
        rows: int,
        cols: int,
        density: float = 0.22,
        preset: str = "Random Obstacles",
        seed: Optional[int] = None,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> "GridMaze":
        """
        Generates a 2D GridMaze matching the requested preset and obstacle density
        that is mathematically guaranteed solvable from start to goal.
        """
        rng = np.random.default_rng(seed)
        s = start if start is not None else (0, 0)
        g = goal if goal is not None else (rows - 1, cols - 1)
        clamped_density = max(0.0, min(0.95, float(density)))
        preset_lower = preset.lower()

        if "open" in preset_lower:
            grid = np.zeros((rows, cols), dtype=np.int32)
        elif "center" in preset_lower:
            grid = np.zeros((rows, cols), dtype=np.int32)
            mid = cols // 2
            grid[0 : max(1, rows - 2), mid] = 1
        elif "snake" in preset_lower:
            grid = np.zeros((rows, cols), dtype=np.int32)
            for r in range(2, rows - 1, 3):
                if (r // 3) % 2 == 0:
                    grid[r, 1:] = 1
                else:
                    grid[r, :-1] = 1
        elif "recursive" in preset_lower or "backtracker" in preset_lower:
            grid = cls._generate_recursive_backtracker(rows, cols, s, rng)
        elif "prim" in preset_lower:
            grid = cls._generate_prims_maze(rows, cols, s, rng)
        else:
            grid = None
            for _ in range(12):
                candidate = (rng.random((rows, cols)) < clamped_density).astype(np.int32)
                candidate[s] = 0
                candidate[g] = 0
                if cls._is_reachable(candidate, s, g):
                    grid = candidate
                    break
            if grid is None:
                base = (rng.random((rows, cols)) < clamped_density).astype(np.int32)
                grid = cls._carve_minimum_path_01_bfs(base, s, g, rng)

        grid[s] = 0
        grid[g] = 0
        if not cls._is_reachable(grid, s, g):
            grid = cls._carve_minimum_path_01_bfs(grid, s, g, rng)

        return cls(grid=grid, start=s, goal=g)


class EightPuzzle(Environment):
    """
    Sliding Tile Puzzle Environment supporting both 8-Puzzle (3x3) and 15-Puzzle (4x4).
    State representation: Immutable tuple of length N*N containing numbers 0..(N*N-1),
    where 0 denotes the blank tile.
    """

    STANDARD_GOAL: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    SPIRAL_GOAL: Tuple[int, ...] = (1, 2, 3, 8, 0, 4, 7, 6, 5)
    STANDARD_GOAL_15: Tuple[int, ...] = tuple(list(range(1, 16)) + [0])

    def __init__(
        self,
        start_state: Tuple[int, ...],
        goal_state: Optional[Tuple[int, ...]] = None,
    ):
        dim = int(math.isqrt(len(start_state)))
        if dim * dim != len(start_state) or dim < 2:
            raise ValueError("Start state length must be a perfect square (e.g. 9 or 16).")
        self.dim = dim
        self.start_state = tuple(start_state)
        if goal_state is not None:
            if len(goal_state) != len(start_state):
                raise ValueError("Goal state must match start state dimensions.")
            self.goal_state = tuple(goal_state)
        else:
            self.goal_state = self.STANDARD_GOAL_15 if dim == 4 else self.STANDARD_GOAL

        self._goal_positions_cache: Dict[Tuple[int, ...], Dict[int, Tuple[int, int]]] = {}

    def get_start_state(self) -> Tuple[int, ...]:
        return self.start_state

    def get_goal_state(self) -> Tuple[int, ...]:
        return self.goal_state

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
            ti = tiles[i]
            for j in range(i + 1, n):
                if ti > tiles[j]:
                    inversions += 1
        return inversions

    @classmethod
    def is_solvable(
        cls,
        state: Tuple[int, ...],
        goal_state: Optional[Tuple[int, ...]] = None,
    ) -> bool:
        """
        Checks solvability for any W x W sliding puzzle using inversion parity:
        - Odd width (3x3): Inv(S) == Inv(G) (mod 2)
        - Even width (4x4): (Inv(S) + blank_row(S)) == (Inv(G) + blank_row(G)) (mod 2)
        """
        dim = int(math.isqrt(len(state)))
        if dim * dim != len(state) or dim < 2:
            return False
        default_goal = cls.STANDARD_GOAL_15 if dim == 4 else cls.STANDARD_GOAL
        target = goal_state if goal_state is not None else default_goal
        if len(target) != len(state):
            return False

        inv_s = cls.count_inversions(state)
        inv_g = cls.count_inversions(target)
        if dim % 2 == 1:
            return (inv_s % 2) == (inv_g % 2)

        row_s = state.index(0) // dim
        row_g = target.index(0) // dim
        return ((inv_s + row_s) % 2) == ((inv_g + row_g) % 2)

    @classmethod
    def generate_random_solvable(
        cls,
        goal_state: Optional[Tuple[int, ...]] = None,
        scramble_moves: int = 18,
        seed: Optional[int] = None,
    ) -> Tuple[int, ...]:
        """
        Generates a guaranteed-solvable randomized 3x3 or 4x4 puzzle state by applying
        non-backtracking legal tile slides from the goal configuration.
        """
        rng = np.random.default_rng(seed)
        target = list(goal_state if goal_state is not None else cls.STANDARD_GOAL)
        dim = int(math.isqrt(len(target)))
        state = list(target)
        prev_zero = -1

        for _ in range(max(1, scramble_moves)):
            z = state.index(0)
            r, c = divmod(z, dim)
            candidates = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < dim and 0 <= nc < dim:
                    idx = nr * dim + nc
                    if idx != prev_zero:
                        candidates.append(idx)
            pick = int(rng.choice(candidates))
            state[z], state[pick] = state[pick], state[z]
            prev_zero = z

        if state == target:
            z = state.index(0)
            r, c = divmod(z, dim)
            swap_idx = ((r + 1) % dim) * dim + c if r < dim - 1 else (r - 1) * dim + c
            state[z], state[swap_idx] = state[swap_idx], state[z]

        return tuple(state)

    def get_neighbors(self, state: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], int]]:
        """
        Generates valid successor states by sliding tiles into the blank position (0).
        Returns list of (neighbor_state, step_cost=1).
        """
        dim = self.dim
        blank_idx = state.index(0)
        r, c = divmod(blank_idx, dim)

        neighbors: List[Tuple[Tuple[int, ...], int]] = []
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < dim and 0 <= nc < dim:
                target_idx = nr * dim + nc
                new_state = list(state)
                new_state[blank_idx], new_state[target_idx] = (
                    new_state[target_idx],
                    new_state[blank_idx],
                )
                neighbors.append((tuple(new_state), 1))

        return neighbors

    def _get_goal_positions(self, target_tuple: Tuple[int, ...]) -> Dict[int, Tuple[int, int]]:
        cached = self._goal_positions_cache.get(target_tuple)
        if cached is None:
            cached = {val: divmod(idx, self.dim) for idx, val in enumerate(target_tuple)}
            self._goal_positions_cache[target_tuple] = cached
        return cached

    def heuristic(
        self,
        state: Tuple[int, ...],
        metric: str = "manhattan",
        target: Optional[Tuple[int, ...]] = None,
    ) -> float:
        """
        Computes heuristic distance from state to target (defaults to goal_state):
        - 'manhattan': Sum of Manhattan distances of all numbered tiles (admissible & consistent).
        - 'euclidean': Sum of Euclidean distances of all numbered tiles.
        """
        target_state = tuple(target) if target is not None else self.goal_state
        goal_positions = self._get_goal_positions(target_state)
        total_dist = 0.0
        dim = self.dim
        use_euclidean = metric == "euclidean"

        for idx, val in enumerate(state):
            if val != 0:
                cur_r, cur_c = divmod(idx, dim)
                goal_r, goal_c = goal_positions[val]
                dr = abs(cur_r - goal_r)
                dc = abs(cur_c - goal_c)
                if use_euclidean:
                    total_dist += math.sqrt(dr * dr + dc * dc)
                else:
                    total_dist += dr + dc

        return float(total_dist)

    def get_stats(self) -> Dict[str, Any]:
        """Extracts descriptive metrics for the 8-puzzle / 15-puzzle board."""
        inversions = self.count_inversions(self.start_state)
        solvable = self.is_solvable(self.start_state, self.goal_state)
        manhattan_h = self.heuristic(self.start_state, metric="manhattan")
        euclidean_h = self.heuristic(self.start_state, metric="euclidean")

        return {
            "dim": int(self.dim),
            "inversion_count": inversions,
            "is_solvable": solvable,
            "initial_manhattan": manhattan_h,
            "initial_euclidean": round(euclidean_h, 3),
        }

    def extract_features(self) -> Dict[str, Any]:
        """Backward compatibility alias for get_stats()."""
        return self.get_stats()
