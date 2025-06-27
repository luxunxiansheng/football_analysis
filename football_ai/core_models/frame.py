from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

# Import specialized domain models
from .player import Player
from .referee import Referee
from .goalkeeper import Goalkeeper
from .ball import Ball
from .field import Field
from .ball_control import BallControl


@dataclass
class CameraMotion:
    """Camera motion information for a frame."""

    x_offset: Optional[float] = None
    y_offset: Optional[float] = None
    rotation: Optional[float] = None
    zoom: Optional[float] = None

    def __bool__(self) -> bool:
        """Check if any camera motion data is present."""
        return any([self.x_offset, self.y_offset, self.rotation, self.zoom])


@dataclass
class ProcessingStatus:
    """Processing status flags for tracking pipeline progress."""

    detection_processed: bool = False
    tracking_processed: bool = False
    transformation_processed: bool = False
    team_assignment_processed: bool = False
    ball_assignment_processed: bool = False
    motion_analysis_processed: bool = False
    rendering_processed: bool = False

    def is_complete(self) -> bool:
        """Check if all processing steps are complete."""
        return all(
            [
                self.detection_processed,
                self.tracking_processed,
                self.transformation_processed,
                self.team_assignment_processed,
                self.ball_assignment_processed,
                self.motion_analysis_processed,
            ]
        )


@dataclass
class Frame:
    """
    Comprehensive frame model that consolidates all frame-related data and functionality.
    This replaces FrameData with a more feature-rich, self-contained class.
    """

    # Core frame identification
    frame_number: int
    timestamp: float
    frame_id: Optional[str] = None  # Optional unique frame identifier

    # Raw frame data
    raw_frame: Optional[np.ndarray] = None  # Original frame as numpy array
    frame_shape: Optional[Tuple[int, int, int]] = None  # (height, width, channels)
    frame_size: Optional[int] = None  # Frame size in bytes

    # Field and environment
    field: Field = dataclass_field(default_factory=Field)
    camera_motion: CameraMotion = dataclass_field(default_factory=CameraMotion)

    # Processing state
    processing_status: ProcessingStatus = dataclass_field(
        default_factory=ProcessingStatus
    )
    processing_time: Optional[float] = None  # Time taken to process this frame
    error_messages: List[str] = dataclass_field(
        default_factory=list
    )  # Any processing errors

    # Object collections - primary data storage
    players: Dict[int, Player] = dataclass_field(
        default_factory=dict
    )  # track_id -> Player
    referees: Dict[int, Referee] = dataclass_field(
        default_factory=dict
    )  # track_id -> Referee
    goalkeepers: Dict[int, Goalkeeper] = dataclass_field(
        default_factory=dict
    )  # track_id -> Goalkeeper
    ball: Optional[Ball] = None
    ball_control: BallControl = dataclass_field(default_factory=BallControl)

    # Frame-level analytics
    frame_quality_score: Optional[float] = None  # Overall frame quality (0-1)
    visibility_conditions: Optional[str] = None  # "clear", "foggy", "dark", etc.
    lighting_conditions: Optional[str] = None  # "daylight", "artificial", "shadows"

    # Match context for this frame
    match_time: Optional[float] = None  # Match time in seconds
    match_period: Optional[str] = None  # "first_half", "second_half", "extra_time"
    score: Optional[Tuple[int, int]] = None  # (home_score, away_score)

    # Additional custom data
    annotations: Dict[str, Any] = dataclass_field(
        default_factory=dict
    )  # Human annotations

    def __post_init__(self):
        """Initialize frame-specific data after creation."""
        if self.raw_frame is not None and self.frame_shape is None:
            shape = self.raw_frame.shape
            if len(shape) == 3:
                self.frame_shape = (shape[0], shape[1], shape[2])
            self.frame_size = self.raw_frame.nbytes

    # Object management methods
    def add_player(self, player: Player) -> None:
        """Add a Player object to the frame."""
        if player.track_id is None:
            raise ValueError("Player must have a track_id")
        self.players[player.track_id] = player

    def add_referee(self, referee: Referee) -> None:
        """Add a Referee object to the frame."""
        if referee.track_id is None:
            raise ValueError("Referee must have a track_id")
        self.referees[referee.track_id] = referee

    def add_goalkeeper(self, goalkeeper: Goalkeeper) -> None:
        """Add a Goalkeeper object to the frame."""
        if goalkeeper.track_id is None:
            raise ValueError("Goalkeeper must have a track_id")
        self.goalkeepers[goalkeeper.track_id] = goalkeeper

    def set_ball(self, ball: Ball) -> None:
        """Set the Ball object for the frame."""
        self.ball = ball

    def remove_player(self, track_id: int) -> Optional[Player]:
        """Remove and return a player by track_id."""
        return self.players.pop(track_id, None)

    def remove_referee(self, track_id: int) -> Optional[Referee]:
        """Remove and return a referee by track_id."""
        return self.referees.pop(track_id, None)

    def remove_goalkeeper(self, track_id: int) -> Optional[Goalkeeper]:
        """Remove and return a goalkeeper by track_id."""
        return self.goalkeepers.pop(track_id, None)

    def clear_ball(self) -> Optional[Ball]:
        """Remove and return the ball object."""
        ball = self.ball
        self.ball = None
        return ball

    # Query methods
    def get_player(self, track_id: int) -> Optional[Player]:
        """Get a specific player by track_id."""
        return self.players.get(track_id)

    def get_referee(self, track_id: int) -> Optional[Referee]:
        """Get a specific referee by track_id."""
        return self.referees.get(track_id)

    def get_goalkeeper(self, track_id: int) -> Optional[Goalkeeper]:
        """Get a specific goalkeeper by track_id."""
        return self.goalkeepers.get(track_id)

    def get_all_players(self) -> List[Union[Player, Goalkeeper]]:
        """Get all players including goalkeepers."""
        return list(self.players.values()) + list(self.goalkeepers.values())

    def get_players_by_team(self, team_id: int) -> List[Union[Player, Goalkeeper]]:
        """Get all players (including goalkeepers) for a specific team."""
        team_players = []
        for player in self.players.values():
            if player.team_id == team_id:
                team_players.append(player)
        for goalkeeper in self.goalkeepers.values():
            if goalkeeper.team_id == team_id:
                team_players.append(goalkeeper)
        return team_players

    def get_all_objects(self) -> List[Union[Player, Goalkeeper, Referee, Ball]]:
        """Get all objects in the frame."""
        objects: List[Union[Player, Goalkeeper, Referee, Ball]] = []
        objects.extend(self.get_all_players())
        objects.extend(self.referees.values())
        if self.ball:
            objects.append(self.ball)
        return objects

    def get_active_objects_count(self) -> Dict[str, int]:
        """Get count of active objects in the frame."""
        return {
            "players": len(self.players),
            "goalkeepers": len(self.goalkeepers),
            "referees": len(self.referees),
            "ball": 1 if self.ball else 0,
            "total": len(self.players)
            + len(self.goalkeepers)
            + len(self.referees)
            + (1 if self.ball else 0),
        }

    # Field-related methods
    def set_field_calibration(
        self,
        corners: List[Tuple[float, float]],
        confidence: float,
        method: str = "manual",
    ) -> None:
        """Set field calibration using the Field object."""
        self.field.set_calibration(corners, confidence, method)

    def is_field_calibrated(self) -> bool:
        """Check if the field is calibrated."""
        return self.field.is_calibrated

    def convert_pixel_to_field(
        self, pixel_pos: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Convert pixel coordinates to field coordinates."""
        return self.field.pixel_to_field(pixel_pos)

    def convert_field_to_pixel(
        self, field_pos: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Convert field coordinates to pixel coordinates."""
        return self.field.field_to_pixel(field_pos)

    def get_tactical_zone(self, field_pos: Tuple[float, float]) -> Optional[str]:
        """Get tactical zone for a field position."""
        return self.field.get_zone_for_position(field_pos)

    def is_in_penalty_area(
        self, field_pos: Tuple[float, float], side: str = "both"
    ) -> bool:
        """Check if position is in penalty area."""
        return self.field.is_in_penalty_area(field_pos, side)

    def is_in_goal_area(
        self, field_pos: Tuple[float, float], side: str = "both"
    ) -> bool:
        """Check if position is in goal area."""
        return self.field.is_in_goal_area(field_pos, side)

    def calculate_field_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Calculate distance between two field positions."""
        return self.field.calculate_distance(pos1, pos2)

    # Ball and possession methods
    def get_ball_control_info(self) -> Optional[Dict[str, Any]]:
        """Get ball control information from the Ball object."""
        if self.ball:
            return {
                "controlling_player": self.ball.controlling_player_id,
                "possession_confidence": self.ball.possession_confidence,
                "possession_team": self.ball.possession_team_id,
                "last_touch_player": self.ball.last_touch_player_id,
                "is_in_play": self.ball.is_in_play,
                "ball_state": self.ball.ball_state,
            }
        return None

    def get_ball_position(self) -> Optional[Tuple[float, float]]:
        """Get ball pixel position."""
        return self.ball.pixel_position if self.ball else None

    def get_ball_field_position(self) -> Optional[Tuple[float, float]]:
        """Get ball field position."""
        return self.ball.field_position if self.ball else None

    # Legacy compatibility methods
    def get_player_positions(self) -> Dict[int, Tuple[float, float]]:
        """Get pixel positions of all players."""
        positions = {}
        for track_id, player in self.players.items():
            if player.pixel_position:
                positions[track_id] = player.pixel_position
        for track_id, goalkeeper in self.goalkeepers.items():
            if goalkeeper.pixel_position:
                positions[track_id] = goalkeeper.pixel_position
        return positions

    def get_player_field_positions(self) -> Dict[int, Tuple[float, float]]:
        """Get field positions of all players."""
        positions = {}
        for track_id, player in self.players.items():
            if player.field_position:
                positions[track_id] = player.field_position
        for track_id, goalkeeper in self.goalkeepers.items():
            if goalkeeper.field_position:
                positions[track_id] = goalkeeper.field_position
        return positions

    def get_team_assignments(self) -> Dict[int, int]:
        """Get team assignments for all players."""
        assignments = {}
        for track_id, player in self.players.items():
            if player.team_id is not None:
                assignments[track_id] = player.team_id
        for track_id, goalkeeper in self.goalkeepers.items():
            if goalkeeper.team_id is not None:
                assignments[track_id] = goalkeeper.team_id
        return assignments

    # Analysis methods
    def calculate_team_centroid(self, team_id: int) -> Optional[Tuple[float, float]]:
        """Calculate the centroid (average position) of a team."""
        team_players = self.get_players_by_team(team_id)
        if not team_players:
            return None

        valid_positions = []
        for player in team_players:
            if player.field_position:
                valid_positions.append(player.field_position)

        if not valid_positions:
            return None

        avg_x = sum(pos[0] for pos in valid_positions) / len(valid_positions)
        avg_y = sum(pos[1] for pos in valid_positions) / len(valid_positions)
        return (avg_x, avg_y)

    def get_formation_string(self, team_id: int) -> Optional[str]:
        """Estimate formation string for a team (e.g., '4-4-2')."""
        team_players = self.get_players_by_team(team_id)
        if len(team_players) < 8:  # Not enough players to determine formation
            return None

        # Basic formation estimation based on field positions
        # This is a simplified implementation
        defenders = midfielders = forwards = 0

        for player in team_players:
            if player.field_position:
                x, y = player.field_position
                if x < self.field.length * 0.3:  # Defensive third
                    defenders += 1
                elif x < self.field.length * 0.7:  # Middle third
                    midfielders += 1
                else:  # Attacking third
                    forwards += 1

        if defenders + midfielders + forwards >= 8:
            return f"{defenders}-{midfielders}-{forwards}"
        return None

    def get_offside_players(
        self, attacking_team_id: int
    ) -> List[Union[Player, Goalkeeper]]:
        """Get list of potentially offside players."""
        # Simplified offside detection
        attacking_players = self.get_players_by_team(attacking_team_id)
        defending_players = []

        # Get defending team players
        for player in self.get_all_players():
            if player.team_id != attacking_team_id and player.team_id is not None:
                defending_players.append(player)

        if not defending_players or not attacking_players:
            return []

        # Find second-to-last defender position
        defender_x_positions = []
        for player in defending_players:
            if player.field_position:
                defender_x_positions.append(player.field_position[0])

        if len(defender_x_positions) < 2:
            return []

        defender_x_positions.sort()
        offside_line = defender_x_positions[1]  # Second-to-last defender

        # Check attacking players
        offside_players = []
        for player in attacking_players:
            if player.field_position and player.field_position[0] > offside_line:
                offside_players.append(player)

        return offside_players

    # Processing and state management
    def mark_processing_complete(self, step: str) -> None:
        """Mark a processing step as complete."""
        if hasattr(self.processing_status, f"{step}_processed"):
            setattr(self.processing_status, f"{step}_processed", True)

    def add_error(self, error_message: str) -> None:
        """Add an error message to the frame."""
        self.error_messages.append(error_message)

    def has_errors(self) -> bool:
        """Check if frame has any errors."""
        return len(self.error_messages) > 0

    def is_processing_complete(self) -> bool:
        """Check if all processing is complete."""
        return self.processing_status.is_complete()

    # Frame info and metadata
    def get_frame_info(self) -> Dict[str, Any]:
        """Get comprehensive frame information."""
        return {
            "frame_number": self.frame_number,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "match_time": self.match_time,
            "match_period": self.match_period,
            "frame_shape": self.frame_shape,
            "frame_size": self.frame_size,
            "processing_time": self.processing_time,
            "quality_score": self.frame_quality_score,
            "visibility": self.visibility_conditions,
            "lighting": self.lighting_conditions,
            "field_calibrated": self.is_field_calibrated(),
            "objects_count": self.get_active_objects_count(),
            "has_errors": self.has_errors(),
            "processing_complete": self.is_processing_complete(),
        }

    def clone(self, include_raw_frame: bool = False) -> "Frame":
        """Create a copy of this frame, optionally including raw frame data."""
        # Create a shallow copy and then handle special cases
        new_frame = Frame(
            frame_number=self.frame_number,
            timestamp=self.timestamp,
            frame_id=self.frame_id,
            raw_frame=self.raw_frame if include_raw_frame else None,
            frame_shape=self.frame_shape,
            frame_size=self.frame_size,
        )

        # Copy other attributes
        new_frame.field = self.field  # Fields are immutable enough to share
        new_frame.camera_motion = self.camera_motion
        new_frame.processing_status = ProcessingStatus(
            **self.processing_status.__dict__
        )
        new_frame.processing_time = self.processing_time
        new_frame.error_messages = self.error_messages.copy()
        new_frame.frame_quality_score = self.frame_quality_score
        new_frame.visibility_conditions = self.visibility_conditions
        new_frame.lighting_conditions = self.lighting_conditions
        new_frame.match_time = self.match_time
        new_frame.match_period = self.match_period
        new_frame.score = self.score
        new_frame.annotations = self.annotations.copy()

        # Copy object collections (shallow copy of the dicts)
        new_frame.players = self.players.copy()
        new_frame.referees = self.referees.copy()
        new_frame.goalkeepers = self.goalkeepers.copy()
        new_frame.ball = self.ball

        return new_frame

    def __str__(self) -> str:
        """String representation of the frame."""
        return f"Frame({self.frame_number}, t={self.timestamp:.3f}s, objects={self.get_active_objects_count()['total']})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Frame(frame_number={self.frame_number}, timestamp={self.timestamp}, players={len(self.players)}, goalkeepers={len(self.goalkeepers)}, referees={len(self.referees)}, ball={'Yes' if self.ball else 'No'})"


# Keep the old FrameData as an alias for backward compatibility during transition
FrameData = Frame
