from football_ai.core_models import Video
from football_ai.core_models.interfaces import Processor
import math
from tqdm import tqdm


class SpeedProcessor(Processor):
    """
    Computes and annotates the speed of each tracked object (Player, Goalkeeper, Referee, Ball)
    in every frame. Speed is stored in the object's .speed attribute in units per second.
    """

    def __init__(self):
        self._previous_positions = {}  # track_id -> (field_position, frame_idx)

    def process(self, data: Video) -> Video:
        fps = getattr(data, "fps", 25)  # Default to 25 if not set

        # Use progress bar for speed calculation
        progress_bar = tqdm(
            enumerate(data.frames),
            total=len(data.frames),
            desc="Speed analysis",
            unit="frames",
        )

        for frame_idx, frame in progress_bar:
            # Process all tracked objects
            for obj in (
                list(frame.players.values())
                + list(frame.goalkeepers.values())
                + list(frame.referees.values())
            ):
                track_id = obj.track_id
                field_position = obj.field_position
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
                obj.speed = speed
                self._previous_positions[track_id] = (field_position, frame_idx)
            # Optionally, process ball speed if needed
            if frame.ball and frame.ball.field_position is not None:
                track_id = frame.ball.track_id
                field_position = frame.ball.field_position
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
                frame.ball.speed = speed
                self._previous_positions[track_id] = (field_position, frame_idx)

        progress_bar.close()
        return data
