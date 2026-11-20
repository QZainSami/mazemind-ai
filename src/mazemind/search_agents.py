"""
Classical Search Algorithms for MazeMind AI.
Implements SearchResult dataclass and SearchAgent hierarchy:
- BFSAgent (Breadth-First Search)
- DFSAgent (Depth-First Search)
- GreedyAgent (Greedy Best-First Search)
- AStarAgent (A* Search)
Course: CSL-325 Artificial Intelligence Lab (Fall 2026).
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
import heapq
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from .environments import Environment


@dataclass
class SearchResult:
    """
    Standardized result object returned by all search algorithms.
    """
    algorithm_name: str
    found: bool
    path: List[Any] = field(default_factory=list)
    path_cost: int = 0
    nodes_expanded: int = 0
    max_frontier_size: int = 0
    execution_time_ms: float = 0.0
    exploration_history: List[Any] = field(default_factory=list)
    heuristic_metric: str = "none"


class SearchAgent(ABC):
    """
    Abstract base class for state-space search agents.
    Provides standard path reconstruction and interface enforcement.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def solve(self, env: Environment) -> SearchResult:
        """Solves the given environment and returns a SearchResult."""
        pass

    @staticmethod
    def reconstruct_path(
        parent_map: Dict[Any, Optional[Any]],
        start: Any,
        goal: Any,
    ) -> List[Any]:
        """
        Reconstructs the solution path from start to goal using the parent map.
        """
        curr = goal
        path: List[Any] = []
        while curr is not None:
            path.append(curr)
            if curr == start:
                break
            curr = parent_map.get(curr)

        # If goal does not lead back to start
        if not path or path[-1] != start:
            return []

        path.reverse()
        return path


class BFSAgent(SearchAgent):
    """
    Breadth-First Search (BFS) Agent.
    Guarantees shortest path in unweighted graphs using a FIFO queue.
    Completeness: Yes (on finite branching factor).
    Optimality: Yes (for uniform step costs).
    """

    def __init__(self, name: str = "BFS"):
        super().__init__(name)

    def solve(self, env: Environment) -> SearchResult:
        start_time = time.perf_counter()
        start = env.get_start_state()

        if env.is_goal_state(start):
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(
                algorithm_name=self.name,
                found=True,
                path=[start],
                path_cost=0,
                nodes_expanded=0,
                max_frontier_size=1,
                execution_time_ms=round(duration_ms, 3),
                exploration_history=[start],
                heuristic_metric="none",
            )

        queue: deque = deque([start])
        visited: Set[Any] = {start}
        parent_map: Dict[Any, Optional[Any]] = {start: None}
        cost_map: Dict[Any, int] = {start: 0}
        exploration_history: List[Any] = []

        nodes_expanded = 0
        max_frontier_size = len(queue)
        found = False
        goal_state: Optional[Any] = None

        while queue:
            max_frontier_size = max(max_frontier_size, len(queue))
            current = queue.popleft()
            nodes_expanded += 1
            exploration_history.append(current)

            if env.is_goal_state(current):
                found = True
                goal_state = current
                break

            for neighbor, step_cost in env.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = current
                    cost_map[neighbor] = cost_map[current] + step_cost
                    queue.append(neighbor)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        path = (
            self.reconstruct_path(parent_map, start, goal_state)
            if found and goal_state is not None
            else []
        )
        path_cost = cost_map.get(goal_state, 0) if found and goal_state is not None else 0

        return SearchResult(
            algorithm_name=self.name,
            found=found,
            path=path,
            path_cost=path_cost,
            nodes_expanded=nodes_expanded,
            max_frontier_size=max_frontier_size,
            execution_time_ms=round(duration_ms, 3),
            exploration_history=exploration_history,
            heuristic_metric="none",
        )


class DFSAgent(SearchAgent):
    """
    Depth-First Search (DFS) Agent.
    Uses an explicit LIFO stack to prevent call-stack limits on large grids.
    Completeness: Yes (with cycle detection in finite spaces).
    Optimality: No.
    """

    def __init__(self, name: str = "DFS"):
        super().__init__(name)

    def solve(self, env: Environment) -> SearchResult:
        start_time = time.perf_counter()
        start = env.get_start_state()

        if env.is_goal_state(start):
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(
                algorithm_name=self.name,
                found=True,
                path=[start],
                path_cost=0,
                nodes_expanded=0,
                max_frontier_size=1,
                execution_time_ms=round(duration_ms, 3),
                exploration_history=[start],
                heuristic_metric="none",
            )

        stack: List[Any] = [start]
        visited: Set[Any] = set()
        parent_map: Dict[Any, Optional[Any]] = {start: None}
        cost_map: Dict[Any, int] = {start: 0}
        exploration_history: List[Any] = []

        nodes_expanded = 0
        max_frontier_size = len(stack)
        found = False
        goal_state: Optional[Any] = None

        while stack:
            max_frontier_size = max(max_frontier_size, len(stack))
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            nodes_expanded += 1
            exploration_history.append(current)

            if env.is_goal_state(current):
                found = True
                goal_state = current
                break

            neighbors = env.get_neighbors(current)
            # Push in reverse so first neighbor is expanded next
            for neighbor, step_cost in reversed(neighbors):
                if neighbor not in visited:
                    if neighbor not in parent_map:
                        parent_map[neighbor] = current
                        cost_map[neighbor] = cost_map[current] + step_cost
                    stack.append(neighbor)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        path = (
            self.reconstruct_path(parent_map, start, goal_state)
            if found and goal_state is not None
            else []
        )
        path_cost = cost_map.get(goal_state, 0) if found and goal_state is not None else 0

        return SearchResult(
            algorithm_name=self.name,
            found=found,
            path=path,
            path_cost=path_cost,
            nodes_expanded=nodes_expanded,
            max_frontier_size=max_frontier_size,
            execution_time_ms=round(duration_ms, 3),
            exploration_history=exploration_history,
            heuristic_metric="none",
        )


class GreedyAgent(SearchAgent):
    """
    Greedy Best-First Search Agent.
    Evaluates nodes strictly by heuristic estimate: f(n) = h(n).
    Uses heapq with an insertion counter to break ties cleanly.
    Completeness: Yes (with visited check in finite spaces).
    Optimality: No.
    """

    def __init__(
        self,
        name: str = "Greedy Best-First",
        heuristic_metric: str = "manhattan",
    ):
        super().__init__(name)
        self.heuristic_metric = heuristic_metric

    def solve(self, env: Environment) -> SearchResult:
        start_time = time.perf_counter()
        start = env.get_start_state()

        if env.is_goal_state(start):
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(
                algorithm_name=self.name,
                found=True,
                path=[start],
                path_cost=0,
                nodes_expanded=0,
                max_frontier_size=1,
                execution_time_ms=round(duration_ms, 3),
                exploration_history=[start],
                heuristic_metric=self.heuristic_metric,
            )

        counter = 0
        h_start = env.heuristic(start, metric=self.heuristic_metric)
        frontier: List[Tuple[float, int, Any]] = [(h_start, counter, start)]
        visited: Set[Any] = set()
        parent_map: Dict[Any, Optional[Any]] = {start: None}
        cost_map: Dict[Any, int] = {start: 0}
        exploration_history: List[Any] = []

        nodes_expanded = 0
        max_frontier_size = len(frontier)
        found = False
        goal_state: Optional[Any] = None

        while frontier:
            max_frontier_size = max(max_frontier_size, len(frontier))
            h_val, _, current = heapq.heappop(frontier)

            if current in visited:
                continue

            visited.add(current)
            nodes_expanded += 1
            exploration_history.append(current)

            if env.is_goal_state(current):
                found = True
                goal_state = current
                break

            for neighbor, step_cost in env.get_neighbors(current):
                if neighbor not in visited:
                    if neighbor not in parent_map:
                        parent_map[neighbor] = current
                        cost_map[neighbor] = cost_map[current] + step_cost
                    counter += 1
                    h_neighbor = env.heuristic(neighbor, metric=self.heuristic_metric)
                    heapq.heappush(frontier, (h_neighbor, counter, neighbor))

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        path = (
            self.reconstruct_path(parent_map, start, goal_state)
            if found and goal_state is not None
            else []
        )
        path_cost = cost_map.get(goal_state, 0) if found and goal_state is not None else 0

        return SearchResult(
            algorithm_name=self.name,
            found=found,
            path=path,
            path_cost=path_cost,
            nodes_expanded=nodes_expanded,
            max_frontier_size=max_frontier_size,
            execution_time_ms=round(duration_ms, 3),
            exploration_history=exploration_history,
            heuristic_metric=self.heuristic_metric,
        )


class AStarAgent(SearchAgent):
    """
    A* Search Agent.
    Evaluates nodes by f(n) = g(n) + h(n).
    Guarantees optimal paths with admissible & consistent heuristics.
    Completeness: Yes.
    Optimality: Yes.
    """

    def __init__(
        self,
        name: str = "A*",
        heuristic_metric: str = "manhattan",
    ):
        super().__init__(name)
        self.heuristic_metric = heuristic_metric

    def solve(self, env: Environment) -> SearchResult:
        start_time = time.perf_counter()
        start = env.get_start_state()

        if env.is_goal_state(start):
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(
                algorithm_name=self.name,
                found=True,
                path=[start],
                path_cost=0,
                nodes_expanded=0,
                max_frontier_size=1,
                execution_time_ms=round(duration_ms, 3),
                exploration_history=[start],
                heuristic_metric=self.heuristic_metric,
            )

        counter = 0
        g_score: Dict[Any, float] = {start: 0.0}
        h_start = env.heuristic(start, metric=self.heuristic_metric)
        f_start = h_start

        frontier: List[Tuple[float, int, Any]] = [(f_start, counter, start)]
        parent_map: Dict[Any, Optional[Any]] = {start: None}
        closed_set: Set[Any] = set()
        exploration_history: List[Any] = []

        nodes_expanded = 0
        max_frontier_size = len(frontier)
        found = False
        goal_state: Optional[Any] = None

        while frontier:
            max_frontier_size = max(max_frontier_size, len(frontier))
            current_f, _, current = heapq.heappop(frontier)

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_expanded += 1
            exploration_history.append(current)

            if env.is_goal_state(current):
                found = True
                goal_state = current
                break

            current_g = g_score[current]

            for neighbor, step_cost in env.get_neighbors(current):
                tentative_g = current_g + step_cost
                if neighbor in closed_set and tentative_g >= g_score.get(neighbor, float("inf")):
                    continue

                if tentative_g < g_score.get(neighbor, float("inf")):
                    parent_map[neighbor] = current
                    g_score[neighbor] = tentative_g
                    counter += 1
                    h_cost = env.heuristic(neighbor, metric=self.heuristic_metric)
                    f_cost = tentative_g + h_cost
                    heapq.heappush(frontier, (f_cost, counter, neighbor))

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        path = (
            self.reconstruct_path(parent_map, start, goal_state)
            if found and goal_state is not None
            else []
        )
        path_cost = int(g_score.get(goal_state, 0)) if found and goal_state is not None else 0

        return SearchResult(
            algorithm_name=self.name,
            found=found,
            path=path,
            path_cost=path_cost,
            nodes_expanded=nodes_expanded,
            max_frontier_size=max_frontier_size,
            execution_time_ms=round(duration_ms, 3),
            exploration_history=exploration_history,
            heuristic_metric=self.heuristic_metric,
        )
