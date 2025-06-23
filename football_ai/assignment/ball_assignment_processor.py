from ..utils import np, create_progress_bar
from ..domain.data_models import VideoData, FrameData
from ..domain.interfaces import Processor


class BallAssignmentProcessor(Processor):
    def __init__(self, max_distance: float = 50.0):
        self.max_distance = max_distance  # in pixels

    def process(self, data: VideoData) -> VideoData:
        # Use progress bar only if processing many frames (>50)
        if len(data.frames) > 50:
            progress_bar = create_progress_bar(
                iterable=data.frames, desc="Ball assignment", unit="frames"
            )
            frame_iterator = progress_bar
        else:
            frame_iterator = data.frames
            progress_bar = None

        for frame_data in frame_iterator:
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
            # Assign each ball to the closest player/goalkeeper within max_distance
            for ball_idx in ball_indices:
                ball_detection = detections[ball_idx]
                ball_center = self._get_center(ball_detection)
                min_dist = float("inf")
                assigned_idx = None
                for p_idx in player_indices:
                    player_detection = detections[p_idx]
                    player_center = self._get_center(player_detection)
                    dist = np.linalg.norm(
                        np.array(ball_center) - np.array(player_center)
                    )
                    if dist < min_dist and dist <= self.max_distance:
                        min_dist = dist
                        assigned_idx = p_idx
                ball_detection.assigned_player = assigned_idx

        if progress_bar:
            progress_bar.close()
        return data

    def _get_center(self, det):
        bbox = det.bbox
        return [(bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2]
