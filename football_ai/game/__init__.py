"""
Game Management Module

This module contains the core game logic and management functionality.
The Game is the central entity around which all football analysis revolves.
"""

from ..core_models.game import (
    Game,
    Team,
    MatchEvent,
    AnalysisSource,
    MatchType,
    MatchStatus,
)
from .game_factory import GameFactory
from .game_manager import GameManager

__all__ = [
    "Game",
    "Team",
    "MatchEvent",
    "AnalysisSource",
    "MatchType",
    "MatchStatus",
    "GameFactory",
    "GameManager",
]
