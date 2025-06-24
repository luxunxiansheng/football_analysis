from tqdm import tqdm
from football_ai.domain import Video
from football_ai.domain.interfaces import Processor


class ObjectMotionProcessor(Processor):
    def process(self, data: Video) -> Video:
        # Use progress bar for object motion processing
        progress_bar = tqdm(data.frames, desc="Object motion", unit="frames")

        for frame_data in progress_bar:
            detections = frame_data.detections or []
            for detection in detections:
                track_id = detection.track_id
                bbox = detection.bbox
                center = ((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2)
                detection.object_position = center

        progress_bar.close()
        return data
