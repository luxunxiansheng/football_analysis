"""
Modern object tracker using ByteTrack.
"""

from typing import List, Dict, Any, Optional
import numpy as np

try:
    import supervision as sv
except ImportError:
    sv = None

from ..domain.interfaces import ObjectTracker
from ..domain.models import Detection, ObjectType


class ModernByteTracker(ObjectTracker):
    """Modern ByteTracker implementation."""

    def __init__(self):
        """Initialize ByteTracker."""
        if sv is None:
            raise ImportError("supervision library required for tracking")
        self.tracker = sv.ByteTrack()

    def track_objects(self, detections: List[Detection]) -> List[Detection]:
        """Track objects and assign track IDs."""
        if not detections:
            return detections

        # Convert detections to supervision format
        trackable_detections = [
            d
            for d in detections
            if d.object_type in [ObjectType.PLAYER, ObjectType.GOALKEEPER]
        ]

        if not trackable_detections:
            return detections

        # Create supervision detections
        boxes = []
        confidences = []
        class_ids = []

        for det in trackable_detections:
            boxes.append([det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2])
            confidences.append(det.confidence)
            class_ids.append(0)  # Use same class ID for all trackable objects

        sv_detections = sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
        )

        # Update tracker
        tracked_detections = self.tracker.update_with_detections(sv_detections)

        # Update original detections with track IDs
        result_detections = []
        trackable_idx = 0

        for detection in detections:
            if detection.object_type in [ObjectType.PLAYER, ObjectType.GOALKEEPER]:
                if trackable_idx < len(tracked_detections.tracker_id):
                    detection.track_id = int(
                        tracked_detections.tracker_id[trackable_idx]
                    )
                trackable_idx += 1
            result_detections.append(detection)

        return result_detections
