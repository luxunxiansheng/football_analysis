from tqdm import tqdm
from football_ai.core_models import Video
from football_ai.core_models.interfaces import Processor


class ObjectMotionProcessor(Processor):
    def process(self, data: Video) -> Video:
        # Use progress bar for object motion processing
        progress_bar = tqdm(data.frames, desc="Object motion", unit="frames")

        for frame_data in progress_bar:
            # Use new Frame model collections instead of legacy detections
            # Process players
            for player in frame_data.players.values():
                if hasattr(player, "pixel_position") and player.pixel_position:
                    player.object_position = player.pixel_position
            # Process goalkeepers
            for goalkeeper in frame_data.goalkeepers.values():
                if hasattr(goalkeeper, "pixel_position") and goalkeeper.pixel_position:
                    goalkeeper.object_position = goalkeeper.pixel_position
            # Process referees
            for referee in frame_data.referees.values():
                if hasattr(referee, "pixel_position") and referee.pixel_position:
                    referee.object_position = referee.pixel_position
            # Process ball
            if (
                frame_data.ball
                and hasattr(frame_data.ball, "pixel_position")
                and frame_data.ball.pixel_position
            ):
                frame_data.ball.object_position = frame_data.ball.pixel_position

        progress_bar.close()
        return data
