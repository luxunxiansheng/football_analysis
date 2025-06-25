"""
Match Analytics Module

This module provides advanced analytics and insights for football matches.
It processes data from multiple sources to generate comprehensive match statistics,
tactical analysis, and performance metrics.
"""

from .match_statistics import MatchStatisticsAnalyzer
from .heatmap_generator import HeatmapGenerator

__all__ = [
    "MatchStatisticsAnalyzer",
    "HeatmapGenerator",
]
