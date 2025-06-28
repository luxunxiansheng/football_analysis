from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class Ball:
    """
    Comprehensive ball model that consolidates all ball-related data.
    """

    # Core identification
    track_id: int
    ball_id: Optional[str] = (
        None  # Optional external ball ID (in case of multiple balls)
    )

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
    velocity_vector: Optional[Tuple[float, float]] = None  # 2D velocity (vx, vy)

    # Detection data
    ball_size: Optional[float] = None  # Estimated ball diameter in pixels
    detection_confidence: Optional[float] = None

    # Tracking data
    track_confidence: Optional[float] = None
    track_age: Optional[int] = None  # Number of frames tracked
    last_seen_frame: Optional[int] = None
    is_active: bool = True
    is_visible: bool = True  # Whether ball is visible/not occluded

    # Position history for analysis
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    field_position_history: List[Tuple[float, float]] = field(default_factory=list)
    speed_history: List[float] = field(default_factory=list)

    # Ball possession and control
    controlling_player_id: Optional[int] = None  # Player currently controlling ball
    last_touch_player_id: Optional[int] = None  # Last player to touch ball
    last_touch_frame: Optional[int] = None  # Frame of last touch
    possession_team_id: Optional[int] = None  # Team currently in possession
    possession_confidence: Optional[float] = None

    # Explicit object position for motion analysis
    object_position: Optional[Tuple[float, float]] = None

    # Ball state analysis
    ball_state: Optional[str] = None  # "loose", "controlled", "in_play", "out_of_play"
    is_in_play: bool = True
    is_airborne: bool = False  # Whether ball is in the air
    estimated_height: Optional[float] = None  # Estimated height when airborne

    # Performance statistics
    total_distance: Optional[float] = None  # Total distance traveled
    max_speed: Optional[float] = None
    avg_speed: Optional[float] = None
    time_in_play: Optional[float] = None
    time_out_of_play: Optional[float] = None

    # Ball events tracking
    touches: List[Dict[str, Any]] = field(default_factory=list)  # All ball touches
    passes: List[Dict[str, Any]] = field(default_factory=list)  # Completed passes
    shots: List[Dict[str, Any]] = field(default_factory=list)  # Shot attempts
    goals: List[Dict[str, Any]] = field(default_factory=list)  # Goals scored

    # Field area tracking
    current_field_zone: Optional[str] = (
        None  # "defensive_third", "middle_third", "attacking_third"
    )
    time_in_zones: Dict[str, float] = field(
        default_factory=dict
    )  # Time spent in each zone

    # Additional custom data
    frame_timestamp: Optional[float] = None
    position_estimation_confidence: Optional[float] = None
    bbox: Optional[Tuple[float, float, float, float]] = (
        None  # [x1, y1, x2, y2] for detection/tracking
    )

    def update_position(
        self,
        pixel_pos: Tuple[float, float],
        field_pos: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Update ball position and add to history."""
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
        """Get current speed of the ball."""
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
        """Update ball statistics with new data."""
        # Update total distance
        if self.total_distance is None:
            self.total_distance = distance_increment
        else:
            self.total_distance += distance_increment

        # Update speed statistics
        self.speed = current_speed
        self.speed_history.append(current_speed)

        if self.max_speed is None or current_speed > self.max_speed:
            self.max_speed = current_speed

        # Update average speed
        self.avg_speed = self.calculate_average_speed()

        # Update timestamp
        if frame_timestamp is not None:
            self.frame_timestamp = frame_timestamp

    # Ball-specific methods
    def set_possession(
        self,
        player_id: Optional[int],
        team_id: Optional[int],
        frame_number: int,
        confidence: Optional[float] = None,
    ) -> None:
        """Set ball possession to a specific player/team."""
        self.controlling_player_id = player_id
        self.possession_team_id = team_id
        self.possession_confidence = confidence

        if player_id is not None:
            self.last_touch_player_id = player_id
            self.last_touch_frame = frame_number

    def add_touch(
        self,
        player_id: int,
        frame_number: int,
        touch_type: str,
        position: Optional[Tuple[float, float]] = None,
        team_id: Optional[int] = None,
    ) -> None:
        """Record a ball touch by a player."""
        touch: Dict[str, Any] = {
            "player_id": player_id,
            "team_id": team_id,
            "frame": frame_number,
            "timestamp": self.frame_timestamp,
            "type": touch_type,  # "pass", "shot", "dribble", "tackle", "clearance"
            "position": position,
        }
        self.touches.append(touch)
        self.last_touch_player_id = player_id
        self.last_touch_frame = frame_number

    def add_pass(
        self,
        passer_id: int,
        receiver_id: Optional[int],
        frame_number: int,
        start_position: Tuple[float, float],
        end_position: Optional[Tuple[float, float]] = None,
        success: bool = True,
        pass_type: str = "normal",
    ) -> None:
        """Record a pass."""
        pass_data: Dict[str, Any] = {
            "passer_id": passer_id,
            "receiver_id": receiver_id,
            "frame": frame_number,
            "timestamp": self.frame_timestamp,
            "start_position": start_position,
            "end_position": end_position,
            "success": success,
            "type": pass_type,  # "short", "long", "cross", "through_ball"
        }
        self.passes.append(pass_data)

    def add_shot(
        self,
        shooter_id: int,
        frame_number: int,
        shot_position: Tuple[float, float],
        target_position: Optional[Tuple[float, float]] = None,
        is_goal: bool = False,
        shot_type: str = "normal",
    ) -> None:
        """Record a shot attempt."""
        shot_data: Dict[str, Any] = {
            "shooter_id": shooter_id,
            "frame": frame_number,
            "timestamp": self.frame_timestamp,
            "shot_position": shot_position,
            "target_position": target_position,
            "is_goal": is_goal,
            "type": shot_type,  # "normal", "header", "volley", "penalty"
            "speed": self.speed,
        }
        self.shots.append(shot_data)

        if is_goal:
            self.goals.append(shot_data)

    def set_ball_state(self, state: str, frame_number: int) -> None:
        """Set the current state of the ball."""
        self.ball_state = state
        self.is_in_play = state in ["loose", "controlled", "in_play"]

    def set_airborne_status(
        self, is_airborne: bool, estimated_height: Optional[float] = None
    ) -> None:
        """Set whether the ball is airborne and its estimated height."""
        self.is_airborne = is_airborne
        self.estimated_height = estimated_height

    def update_field_zone(self, zone: str, time_increment: float) -> None:
        """Update the current field zone and time tracking."""
        self.current_field_zone = zone
        if zone in self.time_in_zones:
            self.time_in_zones[zone] += time_increment
        else:
            self.time_in_zones[zone] = time_increment

    def is_controlled(self) -> bool:
        """Check if ball is currently controlled by a player."""
        return self.controlling_player_id is not None

    def is_loose(self) -> bool:
        """Check if ball is loose (not controlled)."""
        return not self.is_controlled()

    def get_possession_info(self) -> Dict[str, Optional[Union[int, float]]]:
        """Get ball possession information."""
        return {
            "controlling_player_id": self.controlling_player_id,
            "last_touch_player_id": self.last_touch_player_id,
            "possession_team_id": self.possession_team_id,
            "possession_confidence": self.possession_confidence,
            "last_touch_frame": self.last_touch_frame,
        }

    def get_ball_stats(self) -> Dict[str, Union[int, float, bool]]:
        """Get comprehensive ball statistics."""
        return {
            "total_touches": len(self.touches),
            "total_passes": len(self.passes),
            "total_shots": len(self.shots),
            "total_goals": len(self.goals),
            "total_distance": self.total_distance or 0.0,
            "max_speed": self.max_speed or 0.0,
            "avg_speed": self.avg_speed or 0.0,
            "time_in_play": self.time_in_play or 0.0,
            "is_in_play": self.is_in_play,
            "is_airborne": self.is_airborne,
        }

    def get_tracking_info(self) -> Dict[str, Optional[Union[int, float, bool]]]:
        """Get tracking-related information."""
        return {
            "track_id": self.track_id,
            "track_confidence": self.track_confidence,
            "track_age": self.track_age,
            "last_seen_frame": self.last_seen_frame,
            "is_active": self.is_active,
            "is_visible": self.is_visible,
        }

    def add_position_data(
        self,
        timestamp: float,
        x: float,
        y: float,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
    ) -> None:
        """Add position data to the ball."""
        self.pixel_position = (x, y)
        self.position_history.append((x, y))

        # Calculate and update velocity
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.speed = (velocity_x**2 + velocity_y**2) ** 0.5

        self.frame_timestamp = timestamp

    def add_possession_data(
        self,
        timestamp: float,
        player_id: Optional[str] = None,
        team_id: Optional[str] = None,
        confidence: float = 1.0,
    ) -> None:
        """Add possession data to the ball."""
        if player_id:
            self.controlling_player_id = int(player_id) if player_id.isdigit() else None
            self.last_touch_player_id = self.controlling_player_id

        if team_id:
            self.possession_team_id = int(team_id) if team_id.isdigit() else None

        self.possession_confidence = confidence
        self.frame_timestamp = timestamp

        # Add to possession history if it exists
        if not hasattr(self, "possession_history"):
            self.possession_history = []

        self.possession_history.append(
            {
                "timestamp": timestamp,
                "player_id": player_id,
                "team_id": team_id,
                "confidence": confidence,
            }
        )
