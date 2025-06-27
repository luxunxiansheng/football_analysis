"""
Detection to Domain Object Converter

This processor converts the tracked and assigned detections into proper domain objects
(Player, Goalkeeper, Ball) and stores them in the frame's object collections.
"""

from football_ai.core_models.video import Video
from football_ai.core_models.frame import Frame
from football_ai.core_models.player import Player
from football_ai.core_models.goalkeeper import Goalkeeper
from football_ai.core_models.ball import Ball
from football_ai.core_models.interfaces import Processor
from football_ai.utilities import create_progress_bar


class DetectionConverterProcessor(Processor):
    """
    Converts tracked detections into domain objects and stores them in frames.

    This processor is essential for bridging the gap between the detection/tracking
    pipeline and the game-centric analysis. It takes the enriched detections and
    converts them into the proper domain objects that the analysis expects.
    """

    def __init__(self):
        """Initialize the detection converter processor."""
        pass

    def process(self, data: Video) -> Video:
        """
        Process video frames and convert detections to domain objects.

        Args:
            data: Video object with frames containing detections

        Returns:
            Video object with frames containing domain objects
        """
        # Use progress bar for conversion
        progress_bar = create_progress_bar(
            iterable=data.frames, desc="Converting detections to objects", unit="frames"
        )

        for frame_data in progress_bar:
            self._convert_frame_detections(frame_data)

        progress_bar.close()
        return data

    def _convert_frame_detections(self, frame: Frame) -> None:
        """
        Convert detections in a single frame to domain objects.

        Args:
            frame: Frame object containing detections to convert
        """
        # Access detections that were set by earlier processors
        detections = getattr(frame, "detections", None) or []

        if not detections:
            return

        # Clear existing objects to avoid duplicates
        frame.players.clear()
        frame.goalkeepers.clear()
        frame.ball = None

        for detection in detections:
            # Skip detections without track IDs (invalid detections)
            if (
                not hasattr(detection, "track_id")
                or detection.track_id is None
                or detection.track_id == -1
            ):
                continue

            object_type = getattr(detection, "object_type", None)

            if object_type == "player":
                player = self._create_player_from_detection(detection)
                frame.add_player(player)

            elif object_type == "goalkeeper":
                goalkeeper = self._create_goalkeeper_from_detection(detection)
                frame.add_goalkeeper(goalkeeper)

            elif object_type == "ball":
                ball = self._create_ball_from_detection(detection)
                frame.set_ball(ball)

            # Note: We skip referees for now as they're not needed for game analysis

    def _create_player_from_detection(self, detection) -> Player:
        """
        Create a Player object from a detection.

        Args:
            detection: Detection object with tracking and team assignment data

        Returns:
            Player object
        """
        # Extract position from bbox
        bbox = detection.bbox
        pixel_position = (
            (bbox.x1 + bbox.x2) / 2,  # center x
            (bbox.y1 + bbox.y2) / 2,  # center y
        )

        # Extract team assignment from detection metadata
        team_id = None
        if hasattr(detection, "team") and detection.team is not None:
            team_id = detection.team
        elif hasattr(detection, "metadata") and detection.metadata:
            team_id = detection.metadata.get("team_id") or detection.metadata.get(
                "cluster_id"
            )

        # Create player object
        player = Player(
            track_id=detection.track_id,
            pixel_position=pixel_position,
            team_id=team_id,
            detection_confidence=detection.confidence,
            speed=getattr(detection, "speed", None),
            field_position=getattr(detection, "field_position", None),
        )

        return player

    def _create_goalkeeper_from_detection(self, detection) -> Goalkeeper:
        """
        Create a Goalkeeper object from a detection.

        Args:
            detection: Detection object with tracking and team assignment data

        Returns:
            Goalkeeper object
        """
        # Extract position from bbox
        bbox = detection.bbox
        pixel_position = (
            (bbox.x1 + bbox.x2) / 2,  # center x
            (bbox.y1 + bbox.y2) / 2,  # center y
        )

        # Extract team assignment from detection metadata
        team_id = None
        if hasattr(detection, "team") and detection.team is not None:
            team_id = detection.team
        elif hasattr(detection, "metadata") and detection.metadata:
            team_id = detection.metadata.get("team_id") or detection.metadata.get(
                "cluster_id"
            )

        # Create goalkeeper object
        goalkeeper = Goalkeeper(
            track_id=detection.track_id,
            pixel_position=pixel_position,
            team_id=team_id,
            detection_confidence=detection.confidence,
            speed=getattr(detection, "speed", None),
            field_position=getattr(detection, "field_position", None),
        )

        return goalkeeper

    def _create_ball_from_detection(self, detection) -> Ball:
        """
        Create a Ball object from a detection.

        Args:
            detection: Detection object with tracking data

        Returns:
            Ball object
        """
        # Extract position from bbox
        bbox = detection.bbox
        pixel_position = (
            (bbox.x1 + bbox.x2) / 2,  # center x
            (bbox.y1 + bbox.y2) / 2,  # center y
        )

        # Extract possession information from detection metadata
        controlling_player_id = None
        possession_team_id = None
        possession_confidence = None

        if (
            hasattr(detection, "assigned_player")
            and detection.assigned_player is not None
        ):
            # If ball is assigned to a player, extract their info
            # This would need the actual player detection to get team info
            # For now, we'll leave it None and let the possession analyzer handle it
            pass

        if hasattr(detection, "metadata") and detection.metadata:
            controlling_player_id = detection.metadata.get("controlling_player_id")
            possession_team_id = detection.metadata.get("possession_team_id")
            possession_confidence = detection.metadata.get("possession_confidence")

        # Create ball object
        ball = Ball(
            track_id=detection.track_id,
            pixel_position=pixel_position,
            detection_confidence=detection.confidence,
            speed=getattr(detection, "speed", None),
            field_position=getattr(detection, "field_position", None),
            controlling_player_id=controlling_player_id,
            possession_team_id=possession_team_id,
            possession_confidence=possession_confidence,
        )

        return ball
