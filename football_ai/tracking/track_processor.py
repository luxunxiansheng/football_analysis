from ..domain.data_models import VideoData, FrameData
from ..domain.interfaces import Processor
from ..domain.models import Detection
import numpy as np


class TrackProcessor(Processor):
    def __init__(self, max_distance: float = 50.0):
        self.max_distance = max_distance
        self.next_id = 1
        self.prev_detections = []  # List of (track_id, center_x, center_y)

    def process(self, data: VideoData) -> VideoData:
        self.next_id = 1
        self.prev_detections = []
        for frame_data in data.frames:
            detections = frame_data.detections or []
            centers = [self._get_center(det) for det in detections]
            assigned_ids = [-1] * len(detections)
            # Match to previous detections
            for i, center in enumerate(centers):
                min_dist = float("inf")
                min_j = -1
                for j, (track_id, prev_x, prev_y) in enumerate(self.prev_detections):
                    dist = np.linalg.norm(np.array(center) - np.array([prev_x, prev_y]))
                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        min_j = j
                if min_j >= 0:
                    assigned_ids[i] = self.prev_detections[min_j][0]
                else:
                    assigned_ids[i] = self.next_id
                    self.next_id += 1
            # Assign IDs
            for det, tid in zip(detections, assigned_ids):
                det.track_id = tid
            # Update prev_detections for next frame
            self.prev_detections = [
                (det.track_id, *self._get_center(det)) for det in detections
            ]
        return data

    def _get_center(self, det: Detection):
        bbox = det.bbox
        return ((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2)
