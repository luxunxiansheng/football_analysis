"""
Modern Football Analysis System - Core Domain Models

This module defines clean domain models and interfaces for football video analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


class ObjectType(Enum):
    """Types of objects that can be detected in football videos."""

    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"


class FieldEntityType(Enum):
    """Types of field entities (human actors on the field)."""

    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"


class TeamAssignment(Enum):
    """Team assignment identifications."""

    TEAM_1 = 1
    TEAM_2 = 2
    UNKNOWN = 0


@dataclass
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
class FieldEntityState:
    """Complete state information for a field entity (player, goalkeeper, or referee)."""

    track_id: int
    bbox: BoundingBox
    entity_type: FieldEntityType
    team: Optional[TeamAssignment] = None
    team_color: Optional[Tuple[int, int, int]] = None
    has_ball: bool = False
    position: Optional[Tuple[float, float]] = None
    position_adjusted: Optional[Tuple[float, float]] = None
    position_transformed: Optional[Tuple[float, float]] = None
    speed: Optional[float] = None  # km/h
    distance: Optional[float] = None  # meters
    team_assignment_confidence: str = "preliminary"  # "preliminary" or "confirmed"
    keypoints: Optional["PlayerKeypoints"] = None  # Pose keypoints if available

    @property
    def is_player(self) -> bool:
        """Check if this entity is a player (player or goalkeeper)."""
        return self.entity_type in [FieldEntityType.PLAYER, FieldEntityType.GOALKEEPER]

    @property
    def is_goalkeeper(self) -> bool:
        """Check if this entity is a goalkeeper."""
        return self.entity_type == FieldEntityType.GOALKEEPER

    @property
    def is_referee(self) -> bool:
        """Check if this entity is a referee."""
        return self.entity_type == FieldEntityType.REFEREE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility."""
        return {
            "entity_type": self.entity_type.value,
            "bbox": self.bbox.as_list(),
            "team": self.team.value if self.team else None,
            "team_color": self.team_color,
            "has_ball": self.has_ball,
            "position": self.position,
            "position_adjusted": self.position_adjusted,
            "position_transformed": self.position_transformed,
            "speed": self.speed,
            "distance": self.distance,
            "team_assignment_confidence": self.team_assignment_confidence,
            "keypoints": self.keypoints.keypoints if self.keypoints else None,
        }


@dataclass
class MatchAnalysis:
    """Complete analysis results for a football match."""

    field_entity_tracks: Dict[
        int, List[FieldEntityState]
    ]  # All field entities (players, goalkeepers, referees)
    ball_tracks: List[Dict[int, Dict[str, Any]]]
    team_ball_control: np.ndarray
    camera_movement: List[List[float]]
    total_frames: int
    fps: float = 24.0


@dataclass
class PlayerKeypoints:
    """Represents pose keypoints for a detected player."""

    keypoints: List[Tuple[float, float, float]]  # (x, y, confidence) for each keypoint
    bbox: BoundingBox
    track_id: Optional[int] = None
    confidence: float = 1.0

    # Standard COCO pose keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def get_keypoint(self, index: int) -> Optional[Tuple[float, float, float]]:
        """Get specific keypoint by index."""
        if 0 <= index < len(self.keypoints):
            return self.keypoints[index]
        return None

    def get_visible_keypoints(
        self, min_confidence: float = 0.5
    ) -> List[Tuple[float, float, float]]:
        """Get all keypoints above confidence threshold."""
        return [kp for kp in self.keypoints if kp[2] >= min_confidence]

    @property
    def center_of_mass(self) -> Optional[Tuple[float, float]]:
        """Calculate center of mass from visible keypoints."""
        visible = self.get_visible_keypoints()
        if not visible:
            return None

        x = sum(kp[0] for kp in visible) / len(visible)
        y = sum(kp[1] for kp in visible) / len(visible)
        return (x, y)


@dataclass
class TeamFeatures:
    """
    Represents comprehensive team features for identification.
    Flexible container for any type of team distinguishing characteristics.
    """

    team_id: int
    name: str
    features: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def add_feature(
        self, feature_name: str, feature_value: Any, confidence: float = 1.0
    ) -> None:
        """Add a feature to the team's feature set."""
        self.features[feature_name] = {
            "value": feature_value,
            "confidence": confidence,
        }

    def get_feature(self, feature_name: str) -> Optional[Any]:
        """Get a specific feature value."""
        feature_data = self.features.get(feature_name)
        return feature_data["value"] if feature_data else None

    def get_feature_confidence(self, feature_name: str) -> Optional[float]:
        """Get confidence score for a specific feature."""
        feature_data = self.features.get(feature_name)
        return feature_data["confidence"] if feature_data else None

    def has_feature(self, feature_name: str) -> bool:
        """Check if team has a specific feature."""
        return feature_name in self.features

    def get_all_features(self) -> Dict[str, Any]:
        """Get all feature values."""
        return {name: data["value"] for name, data in self.features.items()}

    def merge_features(self, other_features: "TeamFeatures") -> None:
        """Merge features from another TeamFeatures object."""
        for feature_name, feature_data in other_features.features.items():
            if feature_name not in self.features:
                self.features[feature_name] = feature_data
            else:
                # Keep feature with higher confidence
                if (
                    feature_data["confidence"]
                    > self.features[feature_name]["confidence"]
                ):
                    self.features[feature_name] = feature_data
