"""Statistics and analytics data models for football analysis."""

from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Optional, Tuple


@dataclass
class TeamStats:
    """Statistics for a team."""

    team_id: int
    player_count: int = 0
    avg_position: Optional[Tuple[float, float]] = None
    formation: Optional[str] = None
    possession_percentage: Optional[float] = None


@dataclass
class MatchStats:
    """Overall match statistics."""

    total_players: int = 0
    team_stats: Optional[Dict[int, TeamStats]] = dataclass_field(default_factory=dict)
    ball_possession_time: Optional[Dict[int, float]] = dataclass_field(
        default_factory=dict
    )  # team_id -> seconds
    total_distance_covered: Optional[Dict[int, float]] = dataclass_field(
        default_factory=dict
    )  # track_id -> meters
    heatmap_data: Optional[Dict[int, List[Tuple[float, float]]]] = dataclass_field(
        default_factory=dict
    )  # track_id -> positions
