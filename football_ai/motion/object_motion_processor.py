from ..domain.data_models import VideoData
from ..domain.interfaces import Processor



class ObjectMotionProcessor(Processor):
    def process(self, data: VideoData) -> VideoData:
        for frame_data in data.frames:
            detections = frame_data.detections or []
            object_positions = {}
            for detection in detections:
                obj_id = detection.track_id if detection.track_id is not None else id(detection)
                bbox = detection.bbox
                center = [(bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2]
                object_positions[obj_id] = center
            if frame_data.metadata is None:
                frame_data.metadata = {}
            frame_data.metadata["object_positions"] = object_positions
        return data
