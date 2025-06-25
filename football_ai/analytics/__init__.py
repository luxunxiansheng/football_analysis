"""
Match Analytics Module

This module provides advanced analytics and insights for football matches.
It processes data from multiple sources to generate comprehensive match statistics,
tactical analysis, and performance metrics.
"""

from .match_statistics import MatchStatisticsAnalyzer
from .formation_analyzer import FormationAnalyzer
from .heatmap_generator import HeatmapGenerator
from .possession_analytics import PossessionAnalytics
from .performance_metrics import PerformanceMetrics

__all__ = [
    "MatchStatisticsAnalyzer",
    "FormationAnalyzer",
    "HeatmapGenerator",
    "PossessionAnalytics",
    "PerformanceMetrics",
]
