from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from supervision.detection.core import Detections
from supervision.keypoint.core import KeyPoints


@dataclass
class Referee:
    """
    Comprehensive referee model that consolidates all referee-related data.
    """

    # Core identification
    track_id: int
    referee_id: Optional[str] = None  # Optional external referee ID
    referee_type: Optional[str] = None  # "main", "assistant", "fourth_official"

    # Position data
    pixel_position: Optional[Tuple[float, float]] = (
        None  # Current position in pixels (x, y)
    )
    field_position: Optional[Tuple[float, float]] = (
        None  # Current position on field (x, y)
    )

    # Movement data
    speed: Optional[float] = None  # Current speed
    direction: Optional[float] = None  # Movement direction in degrees
    acceleration: Optional[float] = None

    # Detection data
    detection: Optional[Detections] = None  # Single detection from supervision
    keypoints: Optional[KeyPoints] = None
    detection_confidence: Optional[float] = None

    # Tracking data
    track_confidence: Optional[float] = None
    track_age: Optional[int] = None  # Number of frames tracked
    last_seen_frame: Optional[int] = None
    is_active: bool = True

    # Performance statistics
    total_distance: Optional[float] = None  # Total distance covered
    max_speed: Optional[float] = None
    avg_speed: Optional[float] = None

    # Position history for analysis
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    field_position_history: List[Tuple[float, float]] = field(default_factory=list)

    # Referee-specific data
    zone_coverage: Optional[str] = (
        None  # "center", "left_wing", "right_wing", "goal_line"
    )
    is_on_field: bool = True  # Whether referee is currently on the field
    equipment_visible: bool = False  # Flag/whistle/cards visibility

    # Decision tracking (for advanced analysis)
    decisions_made: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Track referee decisions
    last_decision_frame: Optional[int] = None

    # Additional custom data
    custom: Optional[Dict[str, Any]] = field(default_factory=dict)
    frame_timestamp: Optional[float] = None
    position_estimation_confidence: Optional[float] = None

    def update_position(
        self,
        pixel_pos: Tuple[float, float],
        field_pos: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Update referee position and add to history."""
        self.pixel_position = pixel_pos
        self.position_history.append(pixel_pos)

        if field_pos:
            self.field_position = field_pos
            self.field_position_history.append(field_pos)

    def calculate_distance_traveled(self) -> float:
        """Calculate total distance from position history."""
        if len(self.position_history) < 2:
            return 0.0

        total_distance: float = 0.0
        for i in range(1, len(self.position_history)):
            prev_pos: Tuple[float, float] = self.position_history[i - 1]
            curr_pos: Tuple[float, float] = self.position_history[i]
            distance: float = (
                (curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2
            ) ** 0.5
            total_distance += distance

        return total_distance

    def get_current_speed(self) -> Optional[float]:
        """Get current speed of the referee."""
        return self.speed

    def get_position_at_frame(self, frame_index: int) -> Optional[Tuple[float, float]]:
        """Get position at a specific frame from history."""
        if 0 <= frame_index < len(self.position_history):
            return self.position_history[frame_index]
        return None

    def get_field_position_at_frame(
        self, frame_index: int
    ) -> Optional[Tuple[float, float]]:
        """Get field position at a specific frame from history."""
        if 0 <= frame_index < len(self.field_position_history):
            return self.field_position_history[frame_index]
        return None

    def calculate_average_speed(self) -> float:
        """Calculate average speed from total distance and time."""
        if self.total_distance and self.track_age:
            return self.total_distance / self.track_age
        return 0.0

    def update_statistics(
        self,
        distance_increment: float,
        current_speed: float,
        frame_timestamp: Optional[float] = None,
    ) -> None:
        """Update referee statistics with new data."""
        # Update total distance
        if self.total_distance is None:
            self.total_distance = distance_increment
        else:
            self.total_distance += distance_increment

        # Update speed statistics
        self.speed = current_speed
        if self.max_speed is None or current_speed > self.max_speed:
            self.max_speed = current_speed

        # Update average speed
        self.avg_speed = self.calculate_average_speed()

        # Update timestamp
        if frame_timestamp is not None:
            self.frame_timestamp = frame_timestamp

    def is_main_referee(self) -> bool:
        """Check if this is the main referee."""
        return self.referee_type == "main"

    def is_assistant_referee(self) -> bool:
        """Check if this is an assistant referee (linesman)."""
        return self.referee_type == "assistant"

    def is_fourth_official(self) -> bool:
        """Check if this is the fourth official."""
        return self.referee_type == "fourth_official"

    def add_decision(
        self,
        decision_type: str,
        frame_number: int,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Add a referee decision to the tracking history."""
        decision: Dict[str, Any] = {
            "type": decision_type,
            "frame": frame_number,
            "timestamp": self.frame_timestamp,
            "description": description,
            "confidence": confidence,
        }
        self.decisions_made.append(decision)
        self.last_decision_frame = frame_number

    def get_decisions_count(self) -> int:
        """Get total number of decisions made."""
        return len(self.decisions_made)

    def get_decisions_by_type(self, decision_type: str) -> List[Dict[str, Any]]:
        """Get all decisions of a specific type."""
        return [
            decision
            for decision in self.decisions_made
            if decision["type"] == decision_type
        ]

    def get_referee_info(self) -> Dict[str, Optional[Union[str, int, bool]]]:
        """Get referee-specific information."""
        return {
            "referee_id": self.referee_id,
            "referee_type": self.referee_type,
            "zone_coverage": self.zone_coverage,
            "is_on_field": self.is_on_field,
            "equipment_visible": self.equipment_visible,
            "decisions_count": self.get_decisions_count(),
        }

    def get_tracking_info(self) -> Dict[str, Optional[Union[int, float, bool]]]:
        """Get tracking-related information."""
        return {
            "track_id": self.track_id,
            "track_confidence": self.track_confidence,
            "track_age": self.track_age,
            "last_seen_frame": self.last_seen_frame,
            "is_active": self.is_active,
        }
