"""
RendererProcessor: Modular pipeline processor for rendering annotated football video frames.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
from football_ai.domain.interfaces import Processor
from football_ai.domain.data_models import VideoData, FrameData, ObjectType
from football_ai.utils.bbox_utils import calculate_bbox_center


class RendererProcessor(Processor):
    def __init__(
        self, output_path: str, render_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            output_path (str): Path to save the rendered video (e.g., MP4).
            render_config (dict): Optional config for rendering (colors, overlays, etc.).
        """
        self.output_path = output_path
        self.render_config = render_config or {}

    def process(self, video_data: VideoData) -> VideoData:
        """
        Renders visualizations onto frames and saves the output video.
        Args:
            video_data (VideoData): The processed video data with all analysis results.
        Returns:
            VideoData: The same object (no modification), for pipeline chaining.
        """
        if not video_data.frames:
            raise ValueError("No frames to render in VideoData.")

        # Prepare video writer
        height, width = video_data.resolution[1], video_data.resolution[0]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' for MP4 format
        out = cv2.VideoWriter(
            self.output_path, fourcc, video_data.frame_rate, (width, height)
        )

        for frame_data in video_data.frames:
            frame = (
                frame_data.raw_frame.copy()
                if hasattr(frame_data.raw_frame, "copy")
                else np.array(frame_data.raw_frame)
            )
            frame_analysis = frame_data.frame_analysis or {}
            metadata = frame_data.metadata or {}
            detections = frame_data.detections or []

            # Filter detections by object type
            player_detections = []
            referee_detections = []
            ball_detection = None

            for detection in detections:
                if detection.object_type in [ObjectType.PLAYER, ObjectType.GOALKEEPER]:
                    player_detections.append(detection)
                elif detection.object_type == ObjectType.REFEREE:
                    referee_detections.append(detection)
                elif detection.object_type == ObjectType.BALL:
                    # Only use the first ball detection found
                    if ball_detection is None:
                        ball_detection = detection

            team_colors = frame_analysis.get("team_colors", None)
            possession_info = frame_analysis.get("ball_possession", None)
            camera_movement = frame_analysis.get("camera_movement", None)

            rendered = self.render_frame(
                frame,
                player_detections=player_detections,
                ball_detection=ball_detection,
                referee_detections=referee_detections,
                team_colors=team_colors,
                possession_info=possession_info,
                camera_movement=camera_movement,
            )
            out.write(rendered)
        out.release()
        return video_data

    def render_frame(
        self,
        frame,
        player_detections=None,
        ball_detection=None,
        referee_detections=None,
        team_colors=None,
        possession_info=None,
        camera_movement=None,
    ):
        """
        Render all overlays for a single frame.
        Args:
            frame: np.ndarray, the image to annotate
            player_detections: list of Detection for players/goalkeepers (optional)
            ball_detection: single Detection for the ball (optional)
            referee_detections: list of Detection for referees (optional)
            team_colors: dict (optional)
            possession_info: dict (optional)
            camera_movement: any (optional)
        Returns:
            np.ndarray: annotated frame
        """
        annotated_frame = frame.copy()
        if player_detections:
            self._draw_players(annotated_frame, player_detections)
        if ball_detection:
            self._draw_ball(annotated_frame, ball_detection)
        if referee_detections:
            self._draw_referees(annotated_frame, referee_detections)
        # ...add more rendering calls as needed (overlays, etc)...
        return annotated_frame

    def _draw_referees(self, frame, referee_detections):
        for referee in referee_detections:
            bbox = referee.bbox
            # Draw rectangle for referee
            cv2.rectangle(
                frame,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (0, 0, 0),
                2,
            )

            # Draw track ID if available
            if referee.track_id is not None:
                cv2.putText(
                    frame,
                    f"REF-{referee.track_id}",
                    (int(bbox.x1), int(bbox.y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

    def _draw_ball(self, frame, ball_detection):
        """Draw a single ball detection on the frame."""
        if not ball_detection:
            return

        bbox = ball_detection.bbox.as_list()
        color = (0, 0, 255)  # Red color for ball
        y = int(bbox[1])
        x, _ = calculate_bbox_center(bbox)
        x = int(x)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ]
        )
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    def _draw_players(self, frame, player_detections):
        """Draw all player and goalkeeper detections."""
        for detection in player_detections:
            bbox = detection.bbox.as_list()

            # Color based on object type and team
            is_goalkeeper = detection.object_type == ObjectType.GOALKEEPER
            if detection.team == 1:
                color = (
                    (255, 0, 0) if is_goalkeeper else (255, 255, 0)
                )  # Red for GK, Green for team 1
            elif detection.team == 2:
                color = (
                    (0, 0, 255) if is_goalkeeper else (0, 255, 255)
                )  # Red for GK, Blue for team 2
            else:
                color = (
                    (0, 255, 255) if is_goalkeeper else (128, 128, 128)
                )  # Default colors

            # Draw ellipse at bottom of bbox with optional track ID
            y2 = int(bbox[3])
            x_center, _ = calculate_bbox_center(bbox)
            x_center = int(x_center)
            width = int(bbox[2] - bbox[0])  # x2 - x1

            cv2.ellipse(
                frame,
                center=(x_center, y2),
                axes=(int(width), int(0.35 * width)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=color,
                thickness=2,
                lineType=cv2.LINE_4,
            )

            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            if detection.track_id is not None:
                cv2.rectangle(
                    frame,
                    (int(x1_rect), int(y1_rect)),
                    (int(x2_rect), int(y2_rect)),
                    color,
                    cv2.FILLED,
                )

                x1_text = x1_rect + 12
                if detection.track_id > 99:
                    x1_text -= 10

                cv2.putText(
                    frame,
                    f"{detection.track_id}",
                    (int(x1_text), int(y1_rect + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )
