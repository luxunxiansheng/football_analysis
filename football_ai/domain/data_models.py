from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
class BallControl:
    """Ball control information for a frame."""

    controlling_player: Optional[int] = None
    control_confidence: Optional[float] = None
    last_touch_player: Optional[int] = None
    possession_team: Optional[int] = None


@dataclass
class PlayerStats:
    """Statistics for a single player."""

    track_id: int
    position: Optional[Tuple[float, float]] = None
    field_position: Optional[Tuple[float, float]] = None
    speed: Optional[float] = None
    team: Optional[int] = None
    total_distance: Optional[float] = None
    max_speed: Optional[float] = None
    avg_speed: Optional[float] = None


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
    team_stats: Optional[Dict[int, TeamStats]] = field(default_factory=dict)
    ball_possession_time: Optional[Dict[int, float]] = field(
        default_factory=dict
    )  # team_id -> seconds
    total_distance_covered: Optional[Dict[int, float]] = field(
        default_factory=dict
    )  # track_id -> meters
    heatmap_data: Optional[Dict[int, List[Tuple[float, float]]]] = field(
        default_factory=dict
    )  # track_id -> positions


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


# Minimal types for modular pipeline
class ObjectType:
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    def as_list(self):
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Detection:
    """
    Detection for a single object in a frame.
    - bbox: bounding box
    - keypoints: pose or landmark keypoints
    - object_type: e.g. player, ball, referee
    - confidence: detection confidence
    - Direct fields for tracking, positioning, and analysis data
    """

    bbox: BoundingBox
    keypoints: Optional[List[Any]] = field(default_factory=list)
    object_type: Optional[str] = None
    confidence: float = 1.0

    # Tracking information
    track_id: Optional[int] = None

    # Team assignment
    team: Optional[int] = None

    # Position information
    object_position: Optional[Tuple[float, float]] = None  # Pixel coordinates (x, y)
    field_position: Optional[Tuple[float, float]] = (
        None  # Real-world field coordinates (x, y)
    )

    # Movement analysis
    speed: Optional[float] = None  # Speed in appropriate units (e.g., km/h, m/s)

    # Ball assignment
    assigned_player: Optional[int] = None  # For ball detections, ID of assigned player

    # Additional custom data
    custom: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class FieldConfiguration:
    """Configuration for field transformation and coordinate mapping."""

    field_corners: Optional[List[Tuple[float, float]]] = (
        None  # Corner points of the field
    )
    field_dimensions: Optional[Tuple[float, float]] = (
        None  # Real field dimensions (width, height)
    )
    perspective_matrix: Optional[Any] = (
        None  # Transformation matrix for perspective correction
    )
    calibration_confidence: Optional[float] = None
    is_calibrated: bool = False


@dataclass
class TrackingInfo:
    """Tracking information for objects across frames."""

    track_id: Optional[int] = None
    track_confidence: Optional[float] = None
    track_age: Optional[int] = None  # Number of frames this track has existed
    last_seen: Optional[int] = None  # Frame number when last detected
    is_active: bool = True


@dataclass
class FrameData:
    frame_number: int
    timestamp: float
    raw_frame: Any  # e.g., numpy array
    detections: List[Detection] = field(default_factory=list)

    # Structured data objects
    camera_motion: CameraMotion = field(default_factory=CameraMotion)
    ball_control: BallControl = field(default_factory=BallControl)
    processing_status: ProcessingStatus = field(default_factory=ProcessingStatus)
    field_config: FieldConfiguration = field(default_factory=FieldConfiguration)

    # Object positions (frame-level aggregated data) - explicit fields
    player_positions: Optional[Dict[int, Tuple[float, float]]] = field(
        default_factory=dict
    )  # track_id -> (x, y)
    ball_position: Optional[Tuple[float, float]] = None
    referee_positions: Optional[Dict[int, Tuple[float, float]]] = field(
        default_factory=dict
    )

    # Field positions (frame-level aggregated data) - explicit fields
    player_field_positions: Optional[Dict[int, Tuple[float, float]]] = field(
        default_factory=dict
    )  # track_id -> field (x, y)
    ball_field_position: Optional[Tuple[float, float]] = None

    # Team assignments
    team_assignments: Optional[Dict[int, int]] = field(
        default_factory=dict
    )  # track_id -> team_id

    # Speed and movement data
    player_speeds: Optional[Dict[int, float]] = field(
        default_factory=dict
    )  # track_id -> speed
    ball_speed: Optional[float] = None

    # Processing flags
    working_processor_ran: bool = False
    detection_processed: bool = False
    tracking_processed: bool = False
    transformation_processed: bool = False
    assignment_processed: bool = False

    # Additional custom data
    custom: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class VideoData:
    video_path: str
    frame_rate: float
    resolution: tuple
    duration: float
    frames: List[FrameData] = field(default_factory=list)

    # Video file information
    original_path: Optional[str] = None
    file_size: Optional[int] = None
    codec: Optional[str] = None

    # Processing configuration
    config_used: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None

    # Analysis results
    total_detections: Optional[int] = None
    total_tracks: Optional[int] = None
    track_quality_metrics: Optional[Dict[str, float]] = None

    # Additional custom data
    custom: Optional[Dict[str, Any]] = field(default_factory=dict)
