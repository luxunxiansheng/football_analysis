from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class Player:
    """
    Comprehensive player model that consolidates all player-related data.
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

    # Ball interaction
    has_ball: bool = False
    ball_possession_time: Optional[float] = None  # Total time with ball
    last_ball_touch_frame: Optional[int] = None

    # Explicit object position for motion analysis
    object_position: Optional[Tuple[float, float]] = None

    # Role and formation
    position_role: Optional[str] = None  # e.g., "defender", "midfielder", "forward"
    tactical_role: Optional[str] = None  # "defender", "midfielder", etc.
    formation_position: Optional[str] = None  # e.g., "CB", "LB", "CM", "ST"

    # Additional custom data
    frame_timestamp: Optional[float] = None
    team_assignment_confidence: Optional[float] = None
    position_estimation_confidence: Optional[float] = None

    def update_position(
        self,
        pixel_pos: Tuple[float, float],
        field_pos: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Update player position and add to history."""
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
        """Check if player currently has ball possession."""
        return self.has_ball

    def get_current_speed(self) -> Optional[float]:
        """Get current speed of the player."""
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
        """Update player statistics with new data."""
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

    def is_goalkeeper(self) -> bool:
        """Check if player is a goalkeeper based on position role."""
        return self.position_role == "goalkeeper" or self.formation_position == "GK"

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

    def add_position_data(
        self,
        timestamp: float,
        x: float,
        y: float,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
        speed: float = 0.0,
    ) -> None:
        """Add position data to the player."""
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
