from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from supervision.detection.core import Detections
from supervision.keypoint.core import KeyPoints


@dataclass
class Goalkeeper:
    """
    Comprehensive goalkeeper model that consolidates all goalkeeper-related data.
    Inherits player functionality but adds goalkeeper-specific features.
    """

    # Core identification
    track_id: int
    player_id: Optional[str] = None  # Optional external player ID
    jersey_number: Optional[int] = None

    # Team assignment
    team_id: Optional[int] = None

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
    sprint_count: Optional[int] = None

    # Position history for analysis
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    field_position_history: List[Tuple[float, float]] = field(default_factory=list)

    # Ball interaction (inherited from player)
    has_ball: bool = False
    ball_possession_time: Optional[float] = None  # Total time with ball
    last_ball_touch_frame: Optional[int] = None

    # Goalkeeper-specific data
    goal_defended: Optional[str] = None  # "left" or "right" goal
    penalty_area_position: Optional[Tuple[float, float]] = (
        None  # Position within penalty area
    )
    is_in_penalty_area: bool = True  # Whether goalkeeper is in penalty area
    is_in_goal_area: bool = True  # Whether goalkeeper is in 6-yard box

    # Goalkeeper actions
    saves_made: List[Dict[str, Any]] = field(default_factory=list)
    catches_made: List[Dict[str, Any]] = field(default_factory=list)
    punches_made: List[Dict[str, Any]] = field(default_factory=list)
    kicks_made: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Goal kicks, clearances
    throws_made: List[Dict[str, Any]] = field(default_factory=list)

    # Goalkeeper statistics
    shots_faced: int = 0
    saves_count: int = 0
    goals_conceded: int = 0
    clean_sheet: bool = True
    distribution_accuracy: Optional[float] = None  # Pass/throw accuracy

    # Positioning analysis
    average_position_in_goal: Optional[Tuple[float, float]] = None
    furthest_from_goal: Optional[float] = None  # Max distance from goal line
    time_outside_penalty_area: Optional[float] = None

    # Additional custom data
    custom: Optional[Dict[str, Any]] = field(default_factory=dict)
    frame_timestamp: Optional[float] = None
    team_assignment_confidence: Optional[float] = None
    position_estimation_confidence: Optional[float] = None

    def update_position(
        self,
        pixel_pos: Tuple[float, float],
        field_pos: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Update goalkeeper position and add to history."""
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

    def is_in_possession(self) -> bool:
        """Check if goalkeeper currently has ball possession."""
        return self.has_ball

    def get_current_speed(self) -> Optional[float]:
        """Get current speed of the goalkeeper."""
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
        """Update goalkeeper statistics with new data."""
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

    def add_position_data(
        self,
        timestamp: float,
        x: float,
        y: float,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
        speed: float = 0.0,
    ) -> None:
        """Add position data to the goalkeeper."""
        self.pixel_position = (x, y)

        # Initialize position history if it doesn't exist
        if not hasattr(self, "position_history"):
            self.position_history = []

        self.position_history.append((x, y))

        # Update velocity and speed
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.speed = speed if speed > 0 else (velocity_x**2 + velocity_y**2) ** 0.5

        # Update max speed if this is higher
        if self.max_speed is None or (
            self.speed is not None and self.speed > self.max_speed
        ):
            self.max_speed = self.speed

        self.frame_timestamp = timestamp

    # Goalkeeper-specific methods
    def add_save(
        self,
        frame_number: int,
        save_type: str,
        shot_position: Optional[Tuple[float, float]] = None,
        save_difficulty: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Record a save made by the goalkeeper."""
        save: Dict[str, Any] = {
            "frame": frame_number,
            "timestamp": self.frame_timestamp,
            "type": save_type,  # "catch", "punch", "deflection", "dive"
            "shot_position": shot_position,
            "difficulty": save_difficulty,  # "easy", "medium", "hard", "exceptional"
            "confidence": confidence,
        }
        self.saves_made.append(save)
        self.saves_count += 1

    def add_distribution(
        self,
        frame_number: int,
        distribution_type: str,
        target_position: Optional[Tuple[float, float]] = None,
        success: bool = True,
        distance: Optional[float] = None,
    ) -> None:
        """Record a distribution action (kick, throw, pass)."""
        action: Dict[str, Any] = {
            "frame": frame_number,
            "timestamp": self.frame_timestamp,
            "type": distribution_type,  # "goal_kick", "throw", "pass", "punt"
            "target_position": target_position,
            "success": success,
            "distance": distance,
        }

        if distribution_type in ["kick", "goal_kick", "punt"]:
            self.kicks_made.append(action)
        elif distribution_type == "throw":
            self.throws_made.append(action)

    def record_goal_conceded(self, frame_number: int) -> None:
        """Record a goal conceded."""
        self.goals_conceded += 1
        self.clean_sheet = False

    def record_shot_faced(self, frame_number: int) -> None:
        """Record a shot faced by the goalkeeper."""
        self.shots_faced += 1

    def calculate_save_percentage(self) -> float:
        """Calculate save percentage."""
        if self.shots_faced == 0:
            return 0.0
        return (self.saves_count / self.shots_faced) * 100

    def calculate_goals_conceded_per_shot(self) -> float:
        """Calculate goals conceded per shot faced."""
        if self.shots_faced == 0:
            return 0.0
        return self.goals_conceded / self.shots_faced

    def is_defending_left_goal(self) -> bool:
        """Check if goalkeeper is defending the left goal."""
        return self.goal_defended == "left"

    def is_defending_right_goal(self) -> bool:
        """Check if goalkeeper is defending the right goal."""
        return self.goal_defended == "right"

    def get_goalkeeper_stats(self) -> Dict[str, Union[int, float, bool]]:
        """Get comprehensive goalkeeper statistics."""
        return {
            "shots_faced": self.shots_faced,
            "saves_made": self.saves_count,
            "goals_conceded": self.goals_conceded,
            "save_percentage": self.calculate_save_percentage(),
            "clean_sheet": self.clean_sheet,
            "distributions": len(self.kicks_made) + len(self.throws_made),
            "goal_kicks": len(self.kicks_made),
            "throws": len(self.throws_made),
            "distribution_accuracy": self.distribution_accuracy or 0.0,
        }

    def get_team_info(self) -> Dict[str, Optional[Union[int, float]]]:
        """Get team-related information."""
        return {
            "team_id": self.team_id,
            "team_assignment_confidence": self.team_assignment_confidence,
            "jersey_number": self.jersey_number,
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

    def get_positioning_info(
        self,
    ) -> Dict[str, Optional[Union[str, bool, float, Tuple[float, float]]]]:
        """Get goalkeeper positioning information."""
        return {
            "goal_defended": self.goal_defended,
            "is_in_penalty_area": self.is_in_penalty_area,
            "is_in_goal_area": self.is_in_goal_area,
            "penalty_area_position": self.penalty_area_position,
            "average_position_in_goal": self.average_position_in_goal,
            "furthest_from_goal": self.furthest_from_goal,
        }
