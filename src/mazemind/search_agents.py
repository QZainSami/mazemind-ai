"""
Classical Search Algorithms for MazeMind AI.
Implements SearchResult dataclass and SearchAgent hierarchy:
- BFSAgent (Breadth-First Search)
- DFSAgent (Depth-First Search)
- UCSAgent (Uniform Cost Search / Dijkstra)
- GreedyAgent (Greedy Best-First Search)
- AStarAgent (A* Search & Weighted A* with w >= 1.0)
- BidirectionalBFSAgent (Concurrent Forward + Backward BFS)
- BidirectionalAStarAgent (Concurrent Forward + Backward A*)
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
    Optional trace metadata supports open frontier rendering, cell inspection (g, h, f),
    and bidirectional search visualization.
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
    exploration_sides: Optional[List[int]] = None
    meet_state: Optional[Any] = None
    frontier_spans: Optional[Dict[Any, Tuple[int, int]]] = None
    cell_costs: Optional[Dict[Any, Tuple[float, float, float]]] = None


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
    def _track_grid_metrics(env: Environment) -> bool:
        """Only populate full frontier/cell tooltip metadata for compact grid topologies."""
        return hasattr(env, "grid")

    @staticmethod
    def reconstruct_path(
        parent_map: Dict[Any, Optional[Any]],
        start: Any,
        goal: Any,
    ) -> List[Any]:
        """Reconstructs the solution path from start to goal using the parent map."""
        curr = goal
        path: List[Any] = []
        while curr is not None:
            path.append(curr)
            if curr == start:
                break
            curr = parent_map.get(curr)

        if not path or path[-1] != start:
            return []

        path.reverse()
        return path


class BFSAgent(SearchAgent):
    """
    Breadth-First Search (BFS) Agent.
    Explores shallowest unweighted nodes first using a FIFO queue.
    """

    def __init__(self, name: str = "BFS"):
        super().__init__(name)

    def solve(self, env: Environment) -> SearchResult:
        start_time = time.perf_counter()
        start = env.get_start_state()
        track = self._track_grid_metrics(env)

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
                frontier_spans={start: (0, 1)} if track else None,
                cell_costs={start: (0.0, 0.0, 0.0)} if track else None,
            )

        queue: deque = deque([start])
        visited: Set[Any] = {start}
        parent_map: Dict[Any, Optional[Any]] = {start: None}
        cost_map: Dict[Any, int] = {start: 0}
        exploration_history: List[Any] = []

        inf_step = 10**9
        discover_step: Dict[Any, int] = {start: 0} if track else {}
        expand_step: Dict[Any, int] = {}
        cell_costs: Dict[Any, Tuple[float, float, float]] = (
            {start: (0.0, 0.0, 0.0)} if track else {}
        )

        nodes_expanded = 0
        max_frontier_size = 1
        found = False
        goal_state: Optional[Any] = None

        while queue:
            max_frontier_size = max(max_frontier_size, len(queue))
            current = queue.popleft()
            nodes_expanded += 1
            exploration_history.append(current)
            if track and current not in expand_step:
                expand_step[current] = nodes_expanded

            if env.is_goal_state(current):
                found = True
                goal_state = current
                break

            for neighbor, step_cost in env.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = current
                    nc = cost_map[current] + step_cost
                    cost_map[neighbor] = nc
                    queue.append(neighbor)
                    if track:
                        discover_step.setdefault(neighbor, nodes_expanded)
                        cell_costs[neighbor] = (float(nc), 0.0, float(nc))

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        path = (
            self.reconstruct_path(parent_map, start, goal_state)
            if found and goal_state is not None
            else []
        )
        path_cost = cost_map.get(goal_state, 0) if found and goal_state is not None else 0
        spans = (
            {s: (discover_step[s], expand_step.get(s, inf_step)) for s in discover_step}
            if track
            else None
        )

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
            frontier_spans=spans,
            cell_costs=cell_costs if track else None,
        )


class DFSAgent(SearchAgent):
    """
    Depth-First Search (DFS) Agent using an explicit LIFO stack with cycle prevention.
    """

    def __init__(self, name: str = "DFS"):
        super().__init__(name)

    def solve(self, env: Environment) -> SearchResult:
        start_time = time.perf_counter()
        start = env.get_start_state()
        track = self._track_grid_metrics(env)

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
                frontier_spans={start: (0, 1)} if track else None,
                cell_costs={start: (0.0, 0.0, 0.0)} if track else None,
            )

        stack: List[Any] = [start]
        visited: Set[Any] = set()
        parent_map: Dict[Any, Optional[Any]] = {start: None}
        cost_map: Dict[Any, int] = {start: 0}
        exploration_history: List[Any] = []

        inf_step = 10**9
        discover_step: Dict[Any, int] = {start: 0} if track else {}
        expand_step: Dict[Any, int] = {}
        cell_costs: Dict[Any, Tuple[float, float, float]] = (
            {start: (0.0, 0.0, 0.0)} if track else {}
        )

        nodes_expanded = 0
        max_frontier_size = 1
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
            if track and current not in expand_step:
                expand_step[current] = nodes_expanded

            if env.is_goal_state(current):
                found = True
                goal_state = current
                break

            neighbors = env.get_neighbors(current)
            for neighbor, step_cost in reversed(neighbors):
                if neighbor not in visited:
                    if neighbor not in parent_map:
                        parent_map[neighbor] = current
                        nc = cost_map[current] + step_cost
                        cost_map[neighbor] = nc
                        if track:
                            discover_step.setdefault(neighbor, nodes_expanded)
                            cell_costs[neighbor] = (float(nc), 0.0, float(nc))
                    stack.append(neighbor)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        path = (
            self.reconstruct_path(parent_map, start, goal_state)
            if found and goal_state is not None
            else []
        )
        path_cost = cost_map.get(goal_state, 0) if found and goal_state is not None else 0
        spans = (
            {s: (discover_step[s], expand_step.get(s, inf_step)) for s in discover_step}
            if track
            else None
        )

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
            frontier_spans=spans,
            cell_costs=cell_costs if track else None,
        )


class UCSAgent(SearchAgent):
    """
    Uniform Cost Search (UCS / Dijkstra) Agent.
    Expands nodes in strict order of lowest cumulative path cost g(n),
    guaranteeing optimal weighted paths even across Mud/Water terrain.
    """

    def __init__(self, name: str = "UCS (Dijkstra)"):
        super().__init__(name)

    def solve(self, env: Environment) -> SearchResult:
        start_time = time.perf_counter()
        start = env.get_start_state()
        track = self._track_grid_metrics(env)

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
                frontier_spans={start: (0, 1)} if track else None,
                cell_costs={start: (0.0, 0.0, 0.0)} if track else None,
            )

        counter = 0
        g_score: Dict[Any, int] = {start: 0}
        frontier: List[Tuple[int, int, Any]] = [(0, counter, start)]
        parent_map: Dict[Any, Optional[Any]] = {start: None}
        closed_set: Set[Any] = set()
        exploration_history: List[Any] = []

        inf_step = 10**9
        discover_step: Dict[Any, int] = {start: 0} if track else {}
        expand_step: Dict[Any, int] = {}
        cell_costs: Dict[Any, Tuple[float, float, float]] = (
            {start: (0.0, 0.0, 0.0)} if track else {}
        )

        nodes_expanded = 0
        max_frontier_size = 1
        found = False
        goal_state: Optional[Any] = None

        while frontier:
            max_frontier_size = max(max_frontier_size, len(frontier))
            curr_g, _, current = heapq.heappop(frontier)

            if current in closed_set or curr_g > g_score.get(current, 10**12):
                continue

            closed_set.add(current)
            nodes_expanded += 1
            exploration_history.append(current)
            if track and current not in expand_step:
                expand_step[current] = nodes_expanded

            if env.is_goal_state(current):
                found = True
                goal_state = current
                break

            for neighbor, step_cost in env.get_neighbors(current):
                tentative_g = curr_g + step_cost
                if tentative_g < g_score.get(neighbor, 10**12):
                    g_score[neighbor] = tentative_g
                    parent_map[neighbor] = current
                    counter += 1
                    heapq.heappush(frontier, (tentative_g, counter, neighbor))
                    if track:
                        discover_step.setdefault(neighbor, nodes_expanded)
                        cell_costs[neighbor] = (float(tentative_g), 0.0, float(tentative_g))

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        path = (
            self.reconstruct_path(parent_map, start, goal_state)
            if found and goal_state is not None
            else []
        )
        path_cost = g_score.get(goal_state, 0) if found and goal_state is not None else 0
        spans = (
            {s: (discover_step[s], expand_step.get(s, inf_step)) for s in discover_step}
            if track
            else None
        )

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
            frontier_spans=spans,
            cell_costs=cell_costs if track else None,
        )


class GreedyAgent(SearchAgent):
    """
    Greedy Best-First Search Agent: evaluates strictly by f(n) = h(n).
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
        track = self._track_grid_metrics(env)
        h_start = env.heuristic(start, metric=self.heuristic_metric)

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
                frontier_spans={start: (0, 1)} if track else None,
                cell_costs={start: (0.0, round(h_start, 2), round(h_start, 2))} if track else None,
            )

        counter = 0
        frontier: List[Tuple[float, int, Any]] = [(h_start, counter, start)]
        visited: Set[Any] = set()
        parent_map: Dict[Any, Optional[Any]] = {start: None}
        cost_map: Dict[Any, int] = {start: 0}
        exploration_history: List[Any] = []

        inf_step = 10**9
        discover_step: Dict[Any, int] = {start: 0} if track else {}
        expand_step: Dict[Any, int] = {}
        cell_costs: Dict[Any, Tuple[float, float, float]] = (
            {start: (0.0, round(h_start, 2), round(h_start, 2))} if track else {}
        )

        nodes_expanded = 0
        max_frontier_size = 1
        found = False
        goal_state: Optional[Any] = None

        while frontier:
            max_frontier_size = max(max_frontier_size, len(frontier))
            _, _, current = heapq.heappop(frontier)

            if current in visited:
                continue

            visited.add(current)
            nodes_expanded += 1
            exploration_history.append(current)
            if track and current not in expand_step:
                expand_step[current] = nodes_expanded

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
                    if track:
                        discover_step.setdefault(neighbor, nodes_expanded)
                        nc = float(cost_map[neighbor])
                        cell_costs[neighbor] = (nc, round(h_neighbor, 2), round(h_neighbor, 2))

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        path = (
            self.reconstruct_path(parent_map, start, goal_state)
            if found and goal_state is not None
            else []
        )
        path_cost = cost_map.get(goal_state, 0) if found and goal_state is not None else 0
        spans = (
            {s: (discover_step[s], expand_step.get(s, inf_step)) for s in discover_step}
            if track
            else None
        )

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
            frontier_spans=spans,
            cell_costs=cell_costs if track else None,
        )


class AStarAgent(SearchAgent):
    """
    A* and Weighted A* Search Agent: evaluates nodes by f(n) = g(n) + weight * h(n).
    Optimal when weight == 1.0 and h(n) is admissible; trades optimality for speed when weight > 1.0.
    """

    def __init__(
        self,
        name: str = "A*",
        heuristic_metric: str = "manhattan",
        weight: float = 1.0,
    ):
        super().__init__(name)
        self.heuristic_metric = heuristic_metric
        self.weight = max(1.0, float(weight))

    def solve(self, env: Environment) -> SearchResult:
        start_time = time.perf_counter()
        start = env.get_start_state()
        track = self._track_grid_metrics(env)
        h_start = env.heuristic(start, metric=self.heuristic_metric)
        f_start = self.weight * h_start

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
                frontier_spans={start: (0, 1)} if track else None,
                cell_costs={start: (0.0, round(h_start, 2), round(f_start, 2))} if track else None,
            )

        counter = 0
        g_score: Dict[Any, float] = {start: 0.0}
        frontier: List[Tuple[float, float, int, float, Any]] = [
            (f_start, h_start, counter, 0.0, start)
        ]
        parent_map: Dict[Any, Optional[Any]] = {start: None}
        expanded_best_g: Dict[Any, float] = {}
        exploration_history: List[Any] = []

        inf_step = 10**9
        discover_step: Dict[Any, int] = {start: 0} if track else {}
        expand_step: Dict[Any, int] = {}
        cell_costs: Dict[Any, Tuple[float, float, float]] = (
            {start: (0.0, round(h_start, 2), round(f_start, 2))} if track else {}
        )

        nodes_expanded = 0
        max_frontier_size = 1
        found = False
        goal_state: Optional[Any] = None

        while frontier:
            max_frontier_size = max(max_frontier_size, len(frontier))
            _, _, _, curr_g, current = heapq.heappop(frontier)

            if curr_g >= expanded_best_g.get(current, float("inf")):
                continue
            if curr_g > g_score.get(current, float("inf")):
                continue

            expanded_best_g[current] = curr_g
            nodes_expanded += 1
            exploration_history.append(current)
            if track and current not in expand_step:
                expand_step[current] = nodes_expanded

            if env.is_goal_state(current):
                found = True
                goal_state = current
                break

            for neighbor, step_cost in env.get_neighbors(current):
                tentative_g = curr_g + step_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    parent_map[neighbor] = current
                    g_score[neighbor] = tentative_g
                    counter += 1
                    h_cost = env.heuristic(neighbor, metric=self.heuristic_metric)
                    f_cost = tentative_g + self.weight * h_cost
                    heapq.heappush(frontier, (f_cost, h_cost, counter, tentative_g, neighbor))
                    if track:
                        discover_step.setdefault(neighbor, nodes_expanded)
                        cell_costs[neighbor] = (
                            round(tentative_g, 2),
                            round(h_cost, 2),
                            round(f_cost, 2),
                        )

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        path = (
            self.reconstruct_path(parent_map, start, goal_state)
            if found and goal_state is not None
            else []
        )
        path_cost = int(g_score.get(goal_state, 0)) if found and goal_state is not None else 0
        spans = (
            {s: (discover_step[s], expand_step.get(s, inf_step)) for s in discover_step}
            if track
            else None
        )

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
            frontier_spans=spans,
            cell_costs=cell_costs if track else None,
        )


def _compute_path_cost(env: Environment, path: List[Any]) -> int:
    """Computes exact transition cost sum along a reconstructed state path."""
    if len(path) <= 1:
        return 0
    total = 0
    for i in range(len(path) - 1):
        curr = path[i]
        nxt = path[i + 1]
        step = 1
        for nbr, cost in env.get_neighbors(curr):
            if nbr == nxt:
                step = cost
                break
        total += step
    return total


def _reconstruct_bidirectional_path(
    parent_fwd: Dict[Any, Optional[Any]],
    parent_bwd: Dict[Any, Optional[Any]],
    start: Any,
    meet: Any,
    goal: Any,
) -> List[Any]:
    """Stitches start -> meet -> goal from forward and backward parent maps."""
    fwd_part = SearchAgent.reconstruct_path(parent_fwd, start, meet)
    if not fwd_part:
        return []
    bwd_part: List[Any] = []
    curr = parent_bwd.get(meet)
    while curr is not None:
        bwd_part.append(curr)
        if curr == goal:
            break
        curr = parent_bwd.get(curr)
    return fwd_part + bwd_part


class BidirectionalBFSAgent(SearchAgent):
    """
    Bidirectional Breadth-First Search Agent.
    Expands concurrently from Start (side=0) and Goal (side=1) until frontiers collide.
    """

    def __init__(self, name: str = "Bidirectional BFS"):
        super().__init__(name)

    def solve(self, env: Environment) -> SearchResult:
        start_time = time.perf_counter()
        start = env.get_start_state()
        goal = env.get_goal_state()
        track = self._track_grid_metrics(env)

        if start == goal or env.is_goal_state(start):
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
                exploration_sides=[0],
                meet_state=start,
                heuristic_metric="none",
                frontier_spans={start: (0, 1)} if track else None,
                cell_costs={start: (0.0, 0.0, 0.0)} if track else None,
            )

        q_fwd: deque = deque([start])
        q_bwd: deque = deque([goal])
        parent_fwd: Dict[Any, Optional[Any]] = {start: None}
        parent_bwd: Dict[Any, Optional[Any]] = {goal: None}
        depth_fwd: Dict[Any, int] = {start: 0}
        depth_bwd: Dict[Any, int] = {goal: 0}

        exploration_history: List[Any] = []
        exploration_sides: List[int] = []
        inf_step = 10**9
        discover_step: Dict[Any, int] = ({start: 0, goal: 0} if track else {})
        expand_step: Dict[Any, int] = {}
        cell_costs: Dict[Any, Tuple[float, float, float]] = (
            {start: (0.0, 0.0, 0.0), goal: (0.0, 0.0, 0.0)} if track else {}
        )

        nodes_expanded = 0
        max_frontier_size = 2
        meet_state: Optional[Any] = None

        while q_fwd and q_bwd and meet_state is None:
            max_frontier_size = max(max_frontier_size, len(q_fwd) + len(q_bwd))

            if len(q_fwd) <= len(q_bwd):
                # Expand one BFS layer forward from start
                for _ in range(len(q_fwd)):
                    current = q_fwd.popleft()
                    nodes_expanded += 1
                    exploration_history.append(current)
                    exploration_sides.append(0)
                    if track and current not in expand_step:
                        expand_step[current] = nodes_expanded

                    if current in parent_bwd:
                        meet_state = current
                        break

                    for neighbor, _ in env.get_neighbors(current):
                        if neighbor not in parent_fwd:
                            parent_fwd[neighbor] = current
                            d = depth_fwd[current] + 1
                            depth_fwd[neighbor] = d
                            q_fwd.append(neighbor)
                            if track:
                                discover_step.setdefault(neighbor, nodes_expanded)
                                cell_costs[neighbor] = (float(d), 0.0, float(d))
                            if neighbor in parent_bwd:
                                meet_state = neighbor
                                break
                    if meet_state is not None:
                        break
            else:
                # Expand one BFS layer backward from goal
                for _ in range(len(q_bwd)):
                    current = q_bwd.popleft()
                    nodes_expanded += 1
                    exploration_history.append(current)
                    exploration_sides.append(1)
                    if track and current not in expand_step:
                        expand_step[current] = nodes_expanded

                    if current in parent_fwd:
                        meet_state = current
                        break

                    for neighbor, _ in env.get_reverse_neighbors(current):
                        if neighbor not in parent_bwd:
                            parent_bwd[neighbor] = current
                            d = depth_bwd[current] + 1
                            depth_bwd[neighbor] = d
                            q_bwd.append(neighbor)
                            if track:
                                discover_step.setdefault(neighbor, nodes_expanded)
                                cell_costs[neighbor] = (float(d), 0.0, float(d))
                            if neighbor in parent_fwd:
                                meet_state = neighbor
                                break
                    if meet_state is not None:
                        break

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        found = meet_state is not None
        path = (
            _reconstruct_bidirectional_path(parent_fwd, parent_bwd, start, meet_state, goal)
            if found and meet_state is not None
            else []
        )
        path_cost = _compute_path_cost(env, path) if found else 0
        spans = (
            {s: (discover_step[s], expand_step.get(s, inf_step)) for s in discover_step}
            if track
            else None
        )

        return SearchResult(
            algorithm_name=self.name,
            found=found,
            path=path,
            path_cost=path_cost,
            nodes_expanded=nodes_expanded,
            max_frontier_size=max_frontier_size,
            execution_time_ms=round(duration_ms, 3),
            exploration_history=exploration_history,
            exploration_sides=exploration_sides,
            meet_state=meet_state,
            heuristic_metric="none",
            frontier_spans=spans,
            cell_costs=cell_costs if track else None,
        )
