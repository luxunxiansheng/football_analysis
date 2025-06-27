from football_ai.utilities import np, create_progress_bar
from football_ai.core_models import Video, Frame
from football_ai.core_models.interfaces import Processor


class BallAssignmentProcessor(Processor):
    def __init__(self, max_distance: float = 50.0):
        self.max_distance = max_distance  # in pixels

    def process(self, data: Video) -> Video:
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
            self._assign_ball_to_players(frame_data)

        if progress_bar:
            progress_bar.close()
        return data

    def _assign_ball_to_players(self, frame: Frame) -> None:
        """Assign ball to the closest player within max distance."""
        if frame.ball is None:
            return

        # Get all players (including goalkeepers)
        all_players = list(frame.players.values()) + list(frame.goalkeepers.values())

        if not all_players:
            return

        ball_position = frame.ball.pixel_position
        if ball_position is None:
            return

        # Find closest player
        min_distance = float("inf")
        closest_player = None

        for player in all_players:
            if player.pixel_position is None:
                continue

            distance = np.linalg.norm(
                np.array(ball_position) - np.array(player.pixel_position)
            )

            if distance < min_distance and distance <= self.max_distance:
                min_distance = distance
                closest_player = player

        # Update ball possession information
        if closest_player is not None:
            frame.ball.controlling_player_id = closest_player.track_id
            frame.ball.possession_team_id = closest_player.team_id
            frame.ball.possession_confidence = float(
                max(0.0, 1.0 - (min_distance / self.max_distance))
            )
        else:
            # Ball is not controlled by any player
            frame.ball.controlling_player_id = None
            frame.ball.possession_team_id = None
            frame.ball.possession_confidence = 0.0

    def _get_center(self, det):
        bbox = det.bbox
        return [(bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2]
