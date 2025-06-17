from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    - metadata: dict for all extra info (track_id, team, field_position, speed, etc.)
    """

    bbox: BoundingBox
    keypoints: Optional[List[Any]] = field(default_factory=list)
    object_type: Optional[str] = None
    confidence: float = 1.0
    metadata: Optional[dict] = field(default_factory=dict)


@dataclass
class FrameData:
    frame_number: int
    timestamp: float
    raw_frame: Any  # e.g., numpy array
    detections: Optional[List[Detection]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class VideoData:
    video_path: str
    frame_rate: float
    resolution: tuple
    duration: float
    frames: List[FrameData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
