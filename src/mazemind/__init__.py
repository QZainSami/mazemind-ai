"""
MazeMind AI: Intelligent Puzzle Solver & Search Algorithm Visualizer.
Course: CSL-325 Artificial Intelligence Lab (Fall 2026).
Group Members: Samiullah Qureshi, Muhammad Musavir Waseem, Farman Ali.
"""

from .environments import Environment, GridMaze, EightPuzzle
from .search_agents import (
    SearchResult,
    SearchAgent,
    BFSAgent,
    DFSAgent,
    GreedyAgent,
    AStarAgent,
)
from .web_animator import (
    build_maze_animator_html,
    build_puzzle_animator_html,
)

__all__ = [
    "Environment",
    "GridMaze",
    "EightPuzzle",
    "SearchResult",
    "SearchAgent",
    "BFSAgent",
    "DFSAgent",
    "GreedyAgent",
    "AStarAgent",
    "build_maze_animator_html",
    "build_puzzle_animator_html",
]
