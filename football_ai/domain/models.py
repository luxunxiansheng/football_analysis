"""
Modern Football Analysis System - Core Domain Models

This module defines clean domain models and interfaces for football video analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


class ObjectType(Enum):
    """Types of objects that can be detected in football videos."""

    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"


class TeamAssignment(Enum):
    """Team assignment identifications."""

    TEAM_1 = 1
    TEAM_2 = 2
    UNKNOWN = 0


@dataclass
class TeamColor:
    """Represents team color information."""

    id: int
    primary_color: Tuple[int, int, int]
    name: str


@dataclass
class BoundingBox:
    """Represents a bounding box for detected objects."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def foot_position(self) -> Tuple[float, float]:
        """Bottom center of bounding box (foot position for players)."""
        return (self.center_x, self.y2)

    def as_list(self) -> List[float]:
        """Convert to list format [x1, y1, x2, y2] for compatibility."""
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to xyxy tuple format."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Detection:
    """Represents a single object detection."""

    bbox: BoundingBox
    object_type: ObjectType
    track_id: Optional[int] = None
    confidence: float = 1.0

    @property
    def class_name(self) -> str:
        """Get the class name for the detected object."""
        return self.object_type.value


@dataclass
class PlayerState:
    """Complete state information for a player."""

    track_id: int
    bbox: BoundingBox
    team: Optional[TeamAssignment] = None
    team_color: Optional[Tuple[int, int, int]] = None
    has_ball: bool = False
    position: Optional[Tuple[float, float]] = None
    position_adjusted: Optional[Tuple[float, float]] = None
    position_transformed: Optional[Tuple[float, float]] = None
    speed: Optional[float] = None  # km/h
    distance: Optional[float] = None  # meters

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility."""
        return {
            "bbox": self.bbox.as_list(),
            "team": self.team.value if self.team else None,
            "team_color": self.team_color,
            "has_ball": self.has_ball,
            "position": self.position,
            "position_adjusted": self.position_adjusted,
            "position_transformed": self.position_transformed,
            "speed": self.speed,
            "distance": self.distance,
        }


@dataclass
class MatchAnalysis:
    """Complete analysis results for a football match."""

    player_tracks: Dict[int, List[PlayerState]]
    referee_tracks: Dict[int, List[Dict[str, Any]]]
    ball_tracks: List[Dict[int, Dict[str, Any]]]
    team_ball_control: np.ndarray
    camera_movement: List[List[float]]
    total_frames: int
    fps: float = 24.0
