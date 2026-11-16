"""
Genetic Algorithm Level Generator for MazeMind AI.
Implements `LevelEvolver` to procedurally evolve solvable mazes matching
user-selected target difficulty and topological characteristics.
"""

from typing import Callable, List, Optional, Tuple
import numpy as np
import pandas as pd

from .environments import GridMaze
from .search_agents import AStarAgent


class LevelEvolver:
    """
    Genetic Algorithm optimizer that evolves binary grid chromosomes into
    solvable, aesthetically challenging mazes tailored to a target path length.
    """

    def __init__(
        self,
        rows: int = 15,
        cols: int = 15,
        target_path_length: int = 25,
        population_size: int = 30,
        generations: int = 40,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        tournament_size: int = 3,
        elitism_count: int = 2,
        seed: Optional[int] = None,
    ):
        self.rows = rows
        self.cols = cols
        self.total_cells = rows * cols
        self.target_path_length = target_path_length
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.rng = np.random.default_rng(seed)
        self.astar_solver = AStarAgent(heuristic_metric="manhattan")

    def _create_individual(self, initial_wall_prob: float = 0.25) -> np.ndarray:
        """Creates a randomized 1D binary chromosome with locked start and goal."""
        chromosome = (self.rng.random(self.total_cells) < initial_wall_prob).astype(np.int32)
        # Lock start (0, 0) and goal (rows - 1, cols - 1) to 0 (open)
        chromosome[0] = 0
        chromosome[self.total_cells - 1] = 0
        return chromosome

    def _chromosome_to_maze(self, chromosome: np.ndarray) -> GridMaze:
        """Converts a 1D chromosome into a 2D GridMaze instance."""
        grid = chromosome.reshape((self.rows, self.cols)).copy()
        return GridMaze(grid=grid, start=(0, 0), goal=(self.rows - 1, self.cols - 1))

    def evaluate_fitness(self, chromosome: np.ndarray) -> float:
        """
        Evaluates chromosome fitness:
        - If unsolvable: Heavy penalty penalized by distance from reachable cells to goal.
        - If solvable:
          Fitness = 100 - |L_actual - L_target| * 2 + (0.5 * dead_ends)
        """
        maze = self._chromosome_to_maze(chromosome)
        result = self.astar_solver.solve(maze)

        if not result.found:
            # Calculate distance of closest visited cell to goal
            gr, gc = maze.goal
            if result.exploration_history:
                min_dist = min(
                    abs(r - gr) + abs(c - gc) for r, c in result.exploration_history
                )
            else:
                min_dist = gr + gc
            # Closeness penalty gradient
            return float(-1000.0 - (min_dist * 10.0))

        actual_length = result.path_cost
        features = maze.extract_features()
        dead_ends = features["dead_end_count"]

        # Reward proximity to target path length + bonus for dead end branches
        length_diff = abs(actual_length - self.target_path_length)
        fitness = 100.0 - (length_diff * 2.0) + (0.5 * dead_ends)
        return float(fitness)

    def _tournament_selection(
        self, population: List[np.ndarray], fitnesses: List[float]
    ) -> np.ndarray:
        """Selects the best individual among k randomly chosen candidates."""
        selected_indices = self.rng.choice(
            len(population), size=self.tournament_size, replace=False
        )
        best_idx = max(selected_indices, key=lambda idx: fitnesses[idx])
        return population[best_idx].copy()

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs single-point crossover with probability crossover_rate."""
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Random crossover point excluding extremities
        cx_point = self.rng.integers(1, self.total_cells - 1)
        child1 = np.concatenate([parent1[:cx_point], parent2[cx_point:]])
        child2 = np.concatenate([parent2[:cx_point], parent1[cx_point:]])

        # Ensure start and goal remain open
        child1[0] = 0
        child1[-1] = 0
        child2[0] = 0
        child2[-1] = 0

        return child1, child2

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Bit-flip mutation applied cell-by-cell."""
        mutation_mask = self.rng.random(self.total_cells) < self.mutation_rate
        # Invert mutated cells
        chromosome = np.where(mutation_mask, 1 - chromosome, chromosome)
        # Lock start and goal
        chromosome[0] = 0
        chromosome[-1] = 0
        return chromosome

    def evolve(
        self,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> Tuple[GridMaze, pd.DataFrame]:
        """
        Executes the genetic algorithm loop.
        Returns:
            fittest_maze: Best GridMaze instance evolved.
            history_df: DataFrame with ['generation', 'best_fitness', 'avg_fitness'].
        """
        # Initialize population
        population = [
            self._create_individual(initial_wall_prob=0.22)
            for _ in range(self.population_size)
        ]
        history_records = []

        best_individual: Optional[np.ndarray] = None
        best_fitness = float("-inf")

        for gen in range(1, self.generations + 1):
            # Evaluate all individuals
            fitnesses = [self.evaluate_fitness(ind) for ind in population]

            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_fitness = fitnesses[gen_best_idx]
            gen_avg_fitness = float(np.mean(fitnesses))

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()

            history_records.append(
                {
                    "generation": gen,
                    "best_fitness": round(best_fitness, 3),
                    "avg_fitness": round(gen_avg_fitness, 3),
                }
            )

            if progress_callback:
                progress_callback(gen, self.generations, best_fitness, gen_avg_fitness)

            # Sort population by fitness descending for elitism
            sorted_indices = np.argsort(fitnesses)[::-1]
            new_population: List[np.ndarray] = [
                population[idx].copy()
                for idx in sorted_indices[: self.elitism_count]
            ]

            # Generate next generation
            while len(new_population) < self.population_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.population_size:
                    new_population.append(c2)

            population = new_population

        history_df = pd.DataFrame(history_records)

        # Fallback if best individual is still None
        if best_individual is None:
            best_individual = population[0]

        fittest_maze = self._chromosome_to_maze(best_individual)
        return fittest_maze, history_df
