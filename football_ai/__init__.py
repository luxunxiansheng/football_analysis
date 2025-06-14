"""
Football AI System

A clean, modular, and extensible football video analysis system.
"""

from .pipeline import FootballAnalysisPipeline
from .domain.models import (
    Detection,
    PlayerState,
    ObjectType,
    TeamAssignment,
    TeamColor,
    BoundingBox,
    MatchAnalysis,
)

__version__ = "1.0.0"
__author__ = "Football AI Team"

__all__ = [
    "FootballAnalysisPipeline",
    "Detection",
    "PlayerState",
    "ObjectType",
    "TeamAssignment",
    "TeamColor",
    "BoundingBox",
    "MatchAnalysis",
]
