from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..domain.models import Detection


@dataclass
class FrameData:
    frame_number: int
    timestamp: float
    raw_frame: Any  # e.g., numpy array
    detections: Optional[List["Detection"]] = field(default_factory=list)
    keypoints: Optional[List[Dict]] = field(default_factory=list)
    tracks: Optional[List[Dict]] = field(default_factory=list)
    team_assignments: Optional[Dict[int, str]] = field(default_factory=dict)
    analysis_results: Optional[Dict[str, Any]] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class VideoData:
    video_path: str
    frame_rate: float
    resolution: tuple
    duration: float
    frames: List[FrameData] = field(default_factory=list)
    global_analysis: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
