"""
Core interfaces for the modern football analysis system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .models import Detection, PlayerState, BoundingBox, TeamColor


class ObjectDetector(ABC):
    """Interface for object detection in football videos."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame."""
        pass


class ObjectTracker(ABC):
    """Interface for tracking objects across video frames."""

    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update object tracking with new detections."""
        pass


class TeamColorAnalyzer(ABC):
    """Interface for analyzing team colors and assigning players to teams."""

    @abstractmethod
    def analyze_frame_colors(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> Dict[int, TeamColor]:
        """Analyze team colors from player detections in a frame."""
        pass

    @abstractmethod
    def assign_player_team(
        self, frame: np.ndarray, detection: Detection
    ) -> Optional[int]:
        """Assign a player to a team based on their jersey color."""
        pass


class BallPossessionAnalyzer(ABC):
    """Interface for analyzing ball possession."""

    @abstractmethod
    def analyze_possession(
        self, ball_detections: List[Detection], player_states: List[PlayerState]
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
