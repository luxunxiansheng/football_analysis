"""
Football AI System

A clean, modular, and extensible football video analysis system with
comprehensive dataclass-based configuration.
"""

from .config import (
    FootballAIConfig,
    get_default_config,
    get_high_accuracy_config,
    get_fast_processing_config,
    get_broadcast_config,
)

__version__ = "1.0.0"
__author__ = "Football AI Team"

__all__ = [
    "FootballAnalysisPipeline",
    "FootballAIConfig",
    "get_default_config",
    "get_high_accuracy_config",
    "get_fast_processing_config",
    "get_broadcast_config",
]
