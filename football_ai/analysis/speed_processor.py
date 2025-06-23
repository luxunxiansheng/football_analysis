from ..domain.data_models import VideoData
from ..domain.interfaces import Processor
import math
from tqdm import tqdm


class SpeedProcessor(Processor):
    """
    Computes and annotates the speed of each detected object (with a track_id and field_position)
    in every frame. Speed is stored in detection.metadata["speed"] in units per second.
    """

    def __init__(self):
        self._previous_positions = {}  # track_id -> (field_position, frame_idx)

    def process(self, data: VideoData) -> VideoData:
        fps = getattr(data, "fps", 25)  # Default to 25 if not set

        # Use progress bar for speed calculation
        progress_bar = tqdm(
            enumerate(data.frames),
            total=len(data.frames),
            desc="Speed analysis",
            unit="frames",
        )

        for frame_idx, frame_data in progress_bar:
            detections = frame_data.detections or []
            for detection in detections:
                track_id = detection.track_id
                field_position = detection.field_position
                if track_id is None or field_position is None:
                    continue
                prev = self._previous_positions.get(track_id)
                speed = None
                if prev is not None:
                    prev_pos, prev_frame = prev
                    dt = (frame_idx - prev_frame) / fps if fps > 0 else 0
                    if dt > 0:
                        dx = field_position[0] - prev_pos[0]
                        dy = field_position[1] - prev_pos[1]
                        dist = math.hypot(dx, dy)
                        speed = dist / dt
                detection.speed = speed
                self._previous_positions[track_id] = (field_position, frame_idx)

        progress_bar.close()
        return data
