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
    UCSAgent,
    GreedyAgent,
    AStarAgent,
    BidirectionalBFSAgent,
)
from .web_animator import (
    build_maze_animator_html,
    build_puzzle_animator_html,
    build_maze_payload,
    build_puzzle_payload,
    get_maze_component,
    get_puzzle_component,
)

__all__ = [
    "Environment",
    "GridMaze",
    "EightPuzzle",
    "SearchResult",
    "SearchAgent",
    "BFSAgent",
    "DFSAgent",
    "UCSAgent",
    "GreedyAgent",
    "AStarAgent",
    "BidirectionalBFSAgent",
    "build_maze_animator_html",
    "build_puzzle_animator_html",
    "build_maze_payload",
    "build_puzzle_payload",
    "get_maze_component",
    "get_puzzle_component",
]
