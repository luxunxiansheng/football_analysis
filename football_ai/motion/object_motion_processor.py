from ..domain.data_models import VideoData
from ..domain.interfaces import Processor


class ObjectMotionProcessor(Processor):
    def process(self, data: VideoData) -> VideoData:
        for frame_data in data.frames:
            detections = frame_data.detections or []
            for detection in detections:
                if detection.metadata is None:
                    detection.metadata = {}
                track_id = (
                    detection.metadata["track_id"]
                    if "track_id" in detection.metadata
                    else None
                )
                bbox = detection.bbox
                center = [(bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2]
                detection.metadata["object_position"] = center
        return data
