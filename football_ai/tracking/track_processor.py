from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.detection.core import Detections
import numpy as np
from tqdm import tqdm

from ..domain.data_models import VideoData
from ..domain.interfaces import Processor


class TrackProcessor(Processor):
    def __init__(self):
        self.tracker = ByteTrack()

    def process(self, data: VideoData) -> VideoData:
        # Use progress bar for tracking
        progress_bar = tqdm(data.frames, desc="Object tracking", unit="frames")

        for frame_data in progress_bar:
            detections = frame_data.detections or []
            if not detections:
                continue
            boxes = np.array([det.bbox.as_list() for det in detections])
            confidences = np.array([det.confidence for det in detections])
            class_ids = np.array(
                [0 for _ in detections]
            )  # Use 0 for all, or map if needed
            sv_detections = Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids,
            )
            tracked = self.tracker.update_with_detections(sv_detections)
            # Assign track IDs back to detections
            for detection, tid in zip(
                detections, getattr(tracked, "tracker_id", [None] * len(detections))
            ):
                if detection.metadata is None:
                    detection.metadata = {}
                detection.metadata["track_id"] = int(tid) if tid is not None else None

        progress_bar.close()
        return data
