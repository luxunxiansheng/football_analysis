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

        # Get ball control data from video metadata (fallback)
        video_ball_control_data = (video_data.metadata or {}).get("ball_control", {})

        # Debug: Check what video-level data we have
        print(
            f"DEBUG Renderer: Video-level ball control data: {video_ball_control_data}"
        )

        # Store rendered frames back in the original video_data object
        progress_bar = tqdm(video_data.frames, desc="Rendering frames", unit="frames")

        for i, frame_data in enumerate(progress_bar):
            # Get frame-specific ball control data (real-time updates)
            frame_ball_control_data = (frame_data.metadata or {}).get(
                "ball_control", video_ball_control_data
            )
            rendered = self.render_frame(frame_data, frame_ball_control_data)
            frame_data.raw_frame = rendered

        progress_bar.close()
        return video_data

    def render_frame(
        self,
        frame_data: FrameData,
        ball_control_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Render all overlays for a single frame.
        Args:
            frame_data: FrameData object containing the frame and detections
            ball_control_data: Ball control data from video metadata
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
        goalkeeper_detections = []
        ball_detection = None
        for detection in detections:
            if detection.object_type in [ObjectType.PLAYER]:
                player_detections.append(detection)
            elif detection.object_type == ObjectType.GOALKEEPER:
                goalkeeper_detections.append(detection)
            elif detection.object_type == ObjectType.REFEREE:
                referee_detections.append(detection)
            elif detection.object_type == ObjectType.BALL:
                if ball_detection is None:
                    ball_detection = detection

        # Optionally extract overlays from frame_analysis if needed
        annotated_frame = frame.copy()
        if player_detections:
            self._draw_players(annotated_frame, player_detections)
        if goalkeeper_detections:
            # self._draw_goalkeepers(annotated_frame, goalkeeper_detections)
            pass
        if ball_detection:
            self._draw_ball(annotated_frame, ball_detection)
        if referee_detections:
            self._draw_referees(annotated_frame, referee_detections)

        self._draw_ball_control(annotated_frame, ball_control_data or {})
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

    def _draw_goalkeepers(self, frame, goalkeeper_detections):
        """Draw all goalkeeper detections."""
        for detection in goalkeeper_detections:
            bbox = detection.bbox.as_list()
            color = (0, 255, 0)  # Green color for goalkeepers

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
                    f"GK{track_id}",
                    (int(x1_text), int(y1_rect + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
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
            team = (
                detection.metadata["team"]
                if detection.metadata and "team" in detection.metadata
                else None
            )
            if team == 1:
                color = (255, 255, 0)
                # Red for GK, Green for team 1
            elif team == 0:
                color = (0, 255, 255)
                # Red for GK, Blue for team 2
            else:
                color = (128, 128, 128)
                # Default colors

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

            # Draw player speed above the player
            self._draw_player_speed(frame, detection, x_center, int(bbox[1]) - 10)

    def _draw_player_speed(self, frame, detection, x_center, y_position):
        """
        Draw speed information for an individual player.

        Args:
            frame: The frame to draw on
            detection: Player detection object
            x_center: X coordinate for speed text
            y_position: Y coordinate for speed text
        """
        if not detection.metadata:
            return

        # Get speed from player metadata
        speed = detection.metadata.get("speed", None)

        if speed is not None:
            # Format speed text
            speed_text = f"{speed:.1f}km/h"

            # Get text size to center it
            text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = x_center - text_size[0] // 2

            # Draw background for better visibility
            bg_padding = 2
            cv2.rectangle(
                frame,
                (text_x - bg_padding, y_position - text_size[1] - bg_padding),
                (text_x + text_size[0] + bg_padding, y_position + bg_padding),
                (0, 0, 0),
                -1,
            )

            # Draw speed text in white
            cv2.putText(
                frame,
                speed_text,
                (text_x, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_ball_control(self, frame, ball_control_data: Dict[str, Any]):
        """
        Draws ball control information on the frame.
        Args:
            frame (np.ndarray): The frame to draw on.
            ball_control_data (dict): Ball control data containing percentages and counts.
        """
        if not ball_control_data:
            # Debug: Show when no data is available
            cv2.putText(
                frame,
                "No ball control data",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),  # Red color for debug
                2,
            )
            return

        percentages = ball_control_data.get("percentages", {})
        total_frames = ball_control_data.get("total_frames_analyzed", 0)

        # Draw background rectangle for better visibility
        overlay_height = 30 + len(percentages) * 25 + 30
        cv2.rectangle(frame, (5, 5), (350, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (350, overlay_height), (255, 255, 255), 2)

        # draw ball control percentages in the top left corner overlay
        y_offset = 30

        if not percentages:
            cv2.putText(
                frame,
                "No possession detected",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        else:
            for team_id, percentage in percentages.items():
                text = f"Team {team_id}: {percentage:.1f}%"
                cv2.putText(
                    frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                y_offset += 25

        # Display total frames analyzed
        cv2.putText(
            frame,
            f"Total frames with ball: {total_frames}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    def _calculate_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (center_x, center_y)
