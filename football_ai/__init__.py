"""
Football AI System - Game-Centric Architecture

A clean, modular, and extensible football video analysis system with
comprehensive dataclass-based configuration and game-centric design.

The system follows a game-centric approach where all analysis revolves around
Game objects, supporting multiple sources (video, live streams, etc.).
"""

# Core configuration
from .config import (
    FootballAIConfig,
    get_default_config,
    get_high_accuracy_config,
    get_fast_processing_config,
    get_broadcast_config,
)

# Game-centric architecture (RECOMMENDED)
from .game import Game, Team, MatchEvent, AnalysisSource, GameFactory, GameManager

# Video analysis components
from .sources.video import VideoAnalysisProcessor, VideoLoader, VideoPipeline

# Utilities
from .utils.config_factory import ConfigFactory

__version__ = "2.0.0"
__author__ = "Football AI Team"

__all__ = [
    # Core configuration
    "FootballAIConfig",
    "get_default_config",
    "get_high_accuracy_config",
    "get_fast_processing_config",
    "get_broadcast_config",
    "ConfigFactory",
    # Game-centric architecture
    "Game",
    "Team",
    "MatchEvent",
    "AnalysisSource",
    "GameFactory",
    "GameManager",
    # Video analysis
    "VideoAnalysisProcessor",
    "VideoLoader",
    "VideoPipeline",
]
