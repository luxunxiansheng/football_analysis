import numpy as np
from ..domain.data_models import VideoData, FrameData
from ..domain.interfaces import Processor
from ..domain.types import Detection


# Minimal Detection class for modular pipeline
class Detection:
    def __init__(self, bbox, object_type=None, confidence=1.0, track_id=None):
        self.bbox = bbox
        self.object_type = object_type
        self.confidence = confidence
        self.track_id = track_id


class BallAssignmentProcessor(Processor):
    def __init__(self, max_distance: float = 50.0):
        self.max_distance = max_distance  # in pixels

    def process(self, data: VideoData) -> VideoData:
        for frame_data in data.frames:
            detections = frame_data.detections or []
            ball_indices = [
                i
                for i, d in enumerate(detections)
                if getattr(d, "object_type", None) == "ball"
                or getattr(d, "class_name", "").lower() == "ball"
            ]
            player_indices = [
                i
                for i, d in enumerate(detections)
                if getattr(d, "object_type", None) in ("player", "goalkeeper")
                or getattr(d, "class_name", "").lower() in ("player", "goalkeeper")
            ]
            assignments = {}
            # Assign each ball to the closest player/goalkeeper within max_distance
            for ball_idx in ball_indices:
                ball_det = detections[ball_idx]
                ball_center = self._get_center(ball_det)
                min_dist = float("inf")
                assigned_idx = None
                for p_idx in player_indices:
                    player_det = detections[p_idx]
                    player_center = self._get_center(player_det)
                    dist = np.linalg.norm(
                        np.array(ball_center) - np.array(player_center)
                    )
                    if dist < min_dist and dist <= self.max_distance:
                        min_dist = dist
                        assigned_idx = p_idx
                assignments[ball_idx] = assigned_idx
            if frame_data.metadata is None:
                frame_data.metadata = {}
            frame_data.metadata["ball_assignments"] = assignments
        return data

    def _get_center(self, det):
        bbox = det.bbox
        return [(bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2]
