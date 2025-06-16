"""
Core interfaces for the modern football analysis system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .models import (
    Detection,
    FieldEntityState,
    BoundingBox,
    TeamColor,
    PlayerKeypoints,
    TeamAssignment,
    TeamFeatures,
)
from .data_models import VideoData


class ObjectDetector(ABC):
    """Interface for object detection in football videos."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame."""
        pass


class KeypointDetector(ABC):
    """Interface for pose/keypoint detection in football videos."""

    @abstractmethod
    def detect_keypoints(self, frame: np.ndarray) -> List[PlayerKeypoints]:
        """Detect pose keypoints for players in a single frame."""
        pass

    @abstractmethod
    def detect_keypoints_from_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> List[PlayerKeypoints]:
        """Detect pose keypoints for specific player detections."""
        pass


class ObjectTracker(ABC):
    """Interface for tracking objects across video frames."""

    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update object tracking with new detections."""
        pass


class TeamFeatureAnalyzer(ABC):
    """Interface for analyzing team distinguishing features in video frames."""

    @abstractmethod
    def analyze_team_features(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> Optional[Dict[int, TeamFeatures]]:
        """
        Analyze team distinguishing features from player detections in a frame.

        Args:
            frame: Video frame
            detections: List of player detections

        Returns:
            Dictionary mapping team IDs to TeamFeatures objects containing
            various distinguishing characteristics (colors, patterns, logos, etc.)
        """
        pass


class BallPossessionAnalyzer(ABC):
    """Interface for analyzing ball possession."""

    @abstractmethod
    def analyze_possession(
        self, ball_detections: List[Detection], player_states: List[FieldEntityState]
    ) -> Dict[str, Any]:
        """Analyze ball possession for the current frame."""
        pass


class CameraMotionTracker(ABC):
    """Interface for tracking camera motion."""

    @abstractmethod
    def track_movement(self, frame: np.ndarray) -> List[float]:
        """Track camera movement in the current frame."""
        pass


class CoordinateTransformer(ABC):
    """Interface for transforming coordinates."""

    @abstractmethod
    def transform_point(
        self, point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Transform image coordinates to field coordinates."""
        pass


class VideoRenderer(ABC):
    """Interface for rendering annotations on video frames."""

    @abstractmethod
    def render_annotations(
        self, frames: List[np.ndarray], analysis: Any
    ) -> List[np.ndarray]:
        """Render all annotations on video frames."""
        pass


class FieldKeypointDetector(ABC):
    """
    Abstract interface for field keypoint detection algorithms.

    This allows the system to be open for extension (new detection algorithms)
    but closed for modification (existing code doesn't change).
    """

    @abstractmethod
    def detect_keypoints(
        self, frame: np.ndarray, **kwargs
    ) -> Optional[List[List[float]]]:
        """
        Detect field keypoints from a video frame.

        Args:
            frame: Video frame as numpy array
            **kwargs: Algorithm-specific parameters

        Returns:
            List of [x, y] coordinates for field corners if detection successful,
            None if detection failed
        """
        pass

    @abstractmethod
    def get_confidence(self) -> float:
        """
        Get confidence score of the last detection.

        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """
        Get the name of the detection algorithm.

        Returns:
            Algorithm name string
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the detector is ready to perform detection.

        Returns:
            True if ready, False otherwise
        """
        pass


class TeamAssigner(ABC):
    """Interface for assigning players to teams based on established team features."""

    @abstractmethod
    def set_team_features(self, team_features: Dict[int, TeamFeatures]) -> None:
        """Set the established team features for assignment."""
        pass

    @abstractmethod
    def assign_player_team(
        self, frame: np.ndarray, detection: Detection
    ) -> Optional[TeamAssignment]:
        """Assign a player to a team based on their distinguishing features."""
        pass

    @abstractmethod
    def get_player_team_assignment(self, track_id: int) -> Optional[TeamAssignment]:
        """Get cached team assignment for a player by track ID."""
        pass

    @abstractmethod
    def has_team_features(self) -> bool:
        """Check if team features have been established."""
        pass


class Processor(ABC):
    @abstractmethod
    def process(self, data: VideoData) -> VideoData:
        """
        Process the input VideoData and return the result.
        """
        pass
