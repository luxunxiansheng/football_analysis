"""
RendererProcessor: Modular pipeline processor for rendering annotated football video frames.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
from tqdm import tqdm
from football_ai.domain.interfaces import Processor
from football_ai.domain.data_models import VideoData, FrameData, ObjectType


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

        # Store rendered frames back in the original video_data object
        progress_bar = tqdm(video_data.frames, desc="Rendering frames", unit="frames")

        for i, frame_data in enumerate(progress_bar):
            rendered = self.render_frame(frame_data)
            frame_data.raw_frame = rendered

        progress_bar.close()
        return video_data

    def render_frame(
        self,
        frame_data: FrameData,
    ):
        """
        Render all overlays for a single frame.
        Args:
            frame_data: FrameData object containing the frame and detections
        Returns:
            np.ndarray: annotated frame
        """
        frame = (
            frame_data.raw_frame.copy()
            if hasattr(frame_data.raw_frame, "copy")
            else np.array(frame_data.raw_frame)
        )
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
                if ball_detection is None:
                    ball_detection = detection

        # Optionally extract overlays from frame_analysis if needed
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
            track_id = (
                referee.metadata["track_id"]
                if referee.metadata and "track_id" in referee.metadata
                else None
            )
            if track_id is not None:
                cv2.putText(
                    frame,
                    f"REF-{track_id}",
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
        x, _ = self._calculate_bbox_center(bbox)
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
            team = (
                detection.metadata["team"]
                if detection.metadata and "team" in detection.metadata
                else None
            )
            if team == 1:
                color = (
                    (255, 0, 0) if is_goalkeeper else (255, 255, 0)
                )  # Red for GK, Green for team 1
            elif team == 2:
                color = (
                    (0, 0, 255) if is_goalkeeper else (0, 255, 255)
                )  # Red for GK, Blue for team 2
            else:
                color = (
                    (0, 255, 255) if is_goalkeeper else (128, 128, 128)
                )  # Default colors

            # Draw ellipse at bottom of bbox with optional track ID
            y2 = int(bbox[3])
            x_center, _ = self._calculate_bbox_center(bbox)
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

            track_id = (
                detection.metadata["track_id"]
                if detection.metadata and "track_id" in detection.metadata
                else None
            )
            if track_id is not None:
                cv2.rectangle(
                    frame,
                    (int(x1_rect), int(y1_rect)),
                    (int(x2_rect), int(y2_rect)),
                    color,
                    cv2.FILLED,
                )

                x1_text = x1_rect + 12
                if track_id > 99:
                    x1_text -= 10

                cv2.putText(
                    frame,
                    f"{track_id}",
                    (int(x1_text), int(y1_rect + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )

    def _calculate_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (center_x, center_y)
