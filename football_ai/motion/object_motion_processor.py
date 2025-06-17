from tqdm import tqdm
from ..domain.data_models import VideoData
from ..domain.interfaces import Processor


class ObjectMotionProcessor(Processor):
    def process(self, data: VideoData) -> VideoData:
        # Use progress bar for object motion processing
        progress_bar = tqdm(data.frames, desc="Object motion", unit="frames")

        for frame_data in progress_bar:
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

        progress_bar.close()
        return data
