"""
Game Management Module

This module contains the core game logic and management functionality.
The Game is the central entity around which all football analysis revolves.
"""

from ..domain.game import Game, Team, MatchEvent, AnalysisSource, MatchType, MatchStatus
from .factory import GameFactory
from .manager import GameManager

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
