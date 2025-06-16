"""
RendererProcessor: Modular pipeline processor for rendering annotated football video frames.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
from football_ai.domain.interfaces import Processor
from football_ai.domain.data_models import VideoData, FrameData


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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.output_path, fourcc, video_data.frame_rate, (width, height)
        )

        for frame_data in video_data.frames:
            frame = (
                frame_data.raw_frame.copy()
                if hasattr(frame_data.raw_frame, "copy")
                else np.array(frame_data.raw_frame)
            )
            analysis_results = frame_data.analysis_results or {}
            metadata = frame_data.metadata or {}
            player_states = analysis_results.get("player_states", [])
            ball_detections = analysis_results.get("ball_detections", [])
            referee_detections = analysis_results.get("referee_detections", [])
            team_colors = analysis_results.get("team_colors", None)
            possession_info = metadata.get("ball_assignments", None)
            camera_movement = analysis_results.get("camera_movement", None)

            rendered = self.render_frame(
                frame,
                player_states=player_states,
                ball_detections=ball_detections,
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
        player_states=None,
        ball_detections=None,
        referee_detections=None,
        team_colors=None,
        possession_info=None,
        camera_movement=None,
    ):
        """
        Render all overlays for a single frame.
        Args:
            frame: np.ndarray, the image to annotate
            player_states: list of FieldEntityState (optional)
            ball_detections: list of Detection (optional)
            referee_detections: list of Detection (optional)
            team_colors: dict (optional)
            possession_info: dict (optional)
            camera_movement: any (optional)
        Returns:
            np.ndarray: annotated frame
        """
        annotated_frame = frame.copy()
        if player_states:
            self._draw_players_and_goalkeepers(annotated_frame, player_states)
        if ball_detections:
            self._draw_balls(annotated_frame, ball_detections)
        # ...add more rendering calls as needed (referees, overlays, etc)...
        return annotated_frame

    @staticmethod
    def _draw_players_and_goalkeepers(frame, player_states):
        for entity in player_states:
            if hasattr(entity, "is_player") and entity.is_player:
                bbox = entity.bbox
                center = (int(bbox.center_x), int(bbox.center_y))
                axes = (int(bbox.width // 2), int(bbox.height // 2))
                color = (0, 255, 0) if not entity.is_goalkeeper else (0, 255, 255)
                cv2.ellipse(frame, center, axes, 0, 0, 360, color, 2)
                if entity.speed is not None:
                    speed_text = f"{entity.speed:.1f} km/h"
                    cv2.putText(
                        frame,
                        speed_text,
                        (center[0] - 20, center[1] - axes[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

    @staticmethod
    def _draw_balls(frame, ball_detections):
        for ball in ball_detections:
            bbox = ball.bbox
            center = (int(bbox.center_x), int(bbox.center_y))
            size = max(int(bbox.width), int(bbox.height), 12)
            pt1 = (center[0], center[1] - size)
            pt2 = (center[0] - size // 2, center[1] + size // 2)
            pt3 = (center[0] + size // 2, center[1] + size // 2)
            triangle_cnt = np.array([pt1, pt2, pt3])
            cv2.drawContours(frame, [triangle_cnt], 0, (0, 0, 255), -1)
            cv2.polylines(
                frame, [triangle_cnt], isClosed=True, color=(255, 255, 255), thickness=2
            )
