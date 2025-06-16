from dataclasses import dataclass
from typing import Optional


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
    bbox: BoundingBox
    object_type: Optional[str] = None
    confidence: float = 1.0
    track_id: Optional[int] = None
    team: Optional[int] = None
    # Add more fields as needed for your pipeline
