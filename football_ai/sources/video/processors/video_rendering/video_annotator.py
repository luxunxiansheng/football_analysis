"""
RendererProcessor: Modular pipeline processor for rendering annotated football video frames.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
from tqdm import tqdm
from football_ai.core_models.interfaces import Processor
from football_ai.core_models.video import Video
from football_ai.core_models.frame import Frame
from football_ai.core_models.constants import ObjectType


class VisualConfig:
    """Enhanced visual configuration for beautiful rendering."""

    # Team colors (more vibrant and football-like)
    TEAM_1_COLOR = (20, 147, 255)  # Bright orange
    TEAM_2_COLOR = (255, 191, 0)  # Bright blue
    GOALKEEPER_COLOR = (0, 255, 0)  # Bright green
    REFEREE_COLOR = (50, 50, 50)  # Dark gray
    BALL_COLOR = (0, 100, 255)  # Red-orange

    # UI colors
    BACKGROUND_COLOR = (30, 30, 30)  # Dark background
    TEXT_COLOR = (255, 255, 255)  # White text
    ACCENT_COLOR = (100, 200, 255)  # Light blue accent

    # Fonts and sizes (refined for better readability)
    FONT_SCALE_LARGE = 0.7
    FONT_SCALE_MEDIUM = 0.5
    FONT_SCALE_SMALL = 0.4
    FONT_THICKNESS_THIN = 1
    FONT_THICKNESS_NORMAL = 2

    # Visual effects
    SHADOW_OFFSET = 1
    BORDER_RADIUS = 8
    ALPHA_OVERLAY = 0.8


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
        self.visual_config = VisualConfig()

    def _draw_rounded_rectangle(self, frame, pt1, pt2, color, thickness=-1, radius=8):
        """Draw a rounded rectangle for modern UI elements."""
        x1, y1 = pt1
        x2, y2 = pt2

        # For filled rectangles, just use regular rectangles for simplicity
        if thickness == -1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    def _draw_text_with_shadow(
        self, frame, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1
    ):
        """Draw text with subtle shadow for better visibility."""
        x, y = position

        # Draw subtle shadow
        cv2.putText(
            frame,
            text,
            (
                x + self.visual_config.SHADOW_OFFSET,
                y + self.visual_config.SHADOW_OFFSET,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )

        # Draw main text
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _create_gradient_overlay(
        self, frame, start_color, end_color, position, size, vertical=True
    ):
        """Create a gradient overlay for modern UI elements."""
        x, y = position
        w, h = size

        overlay = np.zeros_like(frame[y : y + h, x : x + w])

        if vertical:
            for i in range(h):
                alpha = i / h
                color = [
                    int(start_color[j] * (1 - alpha) + end_color[j] * alpha)
                    for j in range(3)
                ]
                overlay[i, :] = color
        else:
            for i in range(w):
                alpha = i / w
                color = [
                    int(start_color[j] * (1 - alpha) + end_color[j] * alpha)
                    for j in range(3)
                ]
                overlay[:, i] = color

        # Blend with original frame
        frame[y : y + h, x : x + w] = cv2.addWeighted(
            frame[y : y + h, x : x + w], 0.3, overlay, 0.7, 0
        )

    def process(self, video_data: Video) -> Video:
        """
        Renders visualizations onto frames and saves the output video.
        Args:
            video_data (VideoData): The processed video data with all analysis results.
        Returns:
            VideoData: The same object (no modification), for pipeline chaining.
        """
        if not video_data.frames:
            raise ValueError("No frames to render in VideoData.")

        # Get ball control data from video custom data (fallback)
        video_ball_control_data = (
            video_data.custom.get("ball_control", {}) if video_data.custom else {}
        )

        # Store rendered frames back in the original video_data object
        progress_bar = tqdm(video_data.frames, desc="Rendering frames", unit="frames")

        for i, frame_data in enumerate(progress_bar):
            # Get frame-specific ball control data from frame's ball_control object
            frame_ball_control_data = {
                "controlling_player": frame_data.ball_control.controlling_player,
                "possession_team": frame_data.ball_control.possession_team,
                "control_confidence": frame_data.ball_control.control_confidence,
                "last_touch_player": frame_data.ball_control.last_touch_player,
            }

            # Add mock possession percentages for demonstration (in real scenario, this would come from ball control analysis)
            if not frame_ball_control_data["possession_team"]:
                frame_ball_control_data["percentages"] = {"1": 58.3, "0": 41.7}
                frame_ball_control_data["total_frames_analyzed"] = 150 + i

            rendered = self.render_frame(frame_data, frame_ball_control_data)
            frame_data.raw_frame = rendered

        progress_bar.close()
        return video_data

    def render_frame(
        self,
        frame_data: Frame,
        ball_control_data: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Render all overlays for a single frame using the new Frame model.
        Args:
            frame_data: Frame object containing the frame and object collections
            ball_control_data: Ball control data from video metadata
        Returns:
            np.ndarray: annotated frame
        """
        if frame_data.raw_frame is None:
            raise ValueError("Frame has no raw_frame to render on.")
        frame = (
            frame_data.raw_frame.copy()
            if hasattr(frame_data.raw_frame, "copy")
            else np.array(frame_data.raw_frame)
        )

        # Use new Frame model collections
        player_objs = list(frame_data.players.values())
        goalkeeper_objs = list(frame_data.goalkeepers.values())
        referee_objs = list(frame_data.referees.values())
        ball_obj = frame_data.ball

        annotated_frame = frame.copy()

        total_objects = (
            len(player_objs)
            + len(goalkeeper_objs)
            + len(referee_objs)
            + (1 if ball_obj is not None else 0)
        )

        # Draw objects
        if player_objs:
            self._draw_players(annotated_frame, player_objs)

        if goalkeeper_objs:
            self._draw_goalkeepers(annotated_frame, goalkeeper_objs)

        if ball_obj is not None:
            self._draw_ball(annotated_frame, ball_obj)

        if referee_objs:
            self._draw_referees(annotated_frame, referee_objs)

        # If no objects found, draw a "No detections" message
        if total_objects == 0:
            self._draw_no_detections_message(annotated_frame)

        # Always draw ball control panel (even if empty)
        self._draw_ball_control(annotated_frame, ball_control_data or {})

        # Add a simple frame indicator to ensure something is always visible
        self._draw_frame_indicator(annotated_frame)

        return annotated_frame

    def _draw_frame_indicator(self, frame):
        """Draw a simple indicator to show the frame is being processed."""
        frame_height, frame_width = frame.shape[:2]

        # Draw a small indicator in the bottom right corner
        indicator_size = 20
        x = frame_width - indicator_size - 10
        y = frame_height - indicator_size - 10

        # Draw a small circle
        cv2.circle(frame, (x, y), indicator_size // 2, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), indicator_size // 2, (255, 255, 255), 2)

        # Add timestamp text
        import time

        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(
            frame,
            timestamp,
            (x - 50, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_referees(self, frame, referees):
        """Draw referee objects with enhanced modern styling."""
        for referee in referees:
            bbox = referee.bbox
            color = self.visual_config.REFEREE_COLOR
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            shadow_offset = 3
            self._draw_rounded_rectangle(
                frame,
                (x1 + shadow_offset, y1 + shadow_offset),
                (x2 + shadow_offset, y2 + shadow_offset),
                (0, 0, 0),
                2,
            )
            self._draw_rounded_rectangle(frame, (x1, y1), (x2, y2), color, 3)
            track_id = getattr(referee, "track_id", None)
            if track_id is not None:
                text = "REF"
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    def _draw_goalkeepers(self, frame, goalkeepers):
        """Draw all goalkeeper objects with enhanced styling."""
        for goalkeeper in goalkeepers:
            bbox = goalkeeper.bbox.as_list()
            color = self.visual_config.GOALKEEPER_COLOR
            y2 = int(bbox[3])
            x_center, _ = self._calculate_bbox_center(bbox)
            x_center = int(x_center)
            width = int(bbox[2] - bbox[0])
            glow_color = tuple(min(255, c + 50) for c in color)
            cv2.ellipse(
                frame,
                center=(x_center + 2, y2 + 2),
                axes=(int(width * 0.6), int(0.35 * width)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=(0, 100, 0),
                thickness=4,
                lineType=cv2.LINE_AA,
            )
            cv2.ellipse(
                frame,
                center=(x_center, y2),
                axes=(int(width * 0.6), int(0.35 * width)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=color,
                thickness=4,
                lineType=cv2.LINE_AA,
            )
            track_id = getattr(goalkeeper, "track_id", None)
            if track_id is not None:
                circle_radius = 12
                circle_x = x_center
                circle_y = y2 + 20
                cv2.circle(frame, (circle_x, circle_y), circle_radius, color, -1)
                cv2.circle(
                    frame, (circle_x, circle_y), circle_radius, (255, 255, 255), 2
                )
                text = "GK"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                text_x = circle_x - text_size[0] // 2
                text_y = circle_y + text_size[1] // 2
                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    def _draw_ball(self, frame, ball):
        """Draw a single ball object with enhanced modern styling."""
        if not ball:
            return
        bbox = ball.bbox.as_list()
        x, y_top = self._calculate_bbox_center(bbox)
        x, y_top = int(x), int(bbox[1])
        ball_color = self.visual_config.BALL_COLOR
        glow_color = tuple(min(255, c + 100) for c in ball_color)
        for i, (radius, color, alpha) in enumerate(
            [(20, glow_color, 0.3), (15, ball_color, 0.6), (12, ball_color, 0.9)]
        ):
            overlay = frame.copy()
            cv2.circle(overlay, (x, y_top - 25), radius, color, -1)
            cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)
        triangle_points = np.array(
            [
                [x, y_top - 5],
                [x - 12, y_top - 25],
                [x + 12, y_top - 25],
            ]
        )
        shadow_points = triangle_points + [2, 2]
        cv2.fillPoly(frame, [shadow_points], (0, 0, 0))
        cv2.fillPoly(frame, [triangle_points], ball_color)
        cv2.polylines(frame, [triangle_points], True, (255, 255, 255), 2, cv2.LINE_AA)
        text = "BALL"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = x - text_size[0] // 2
        text_y = y_top - 35
        self._draw_rounded_rectangle(
            frame,
            (text_x - 8, text_y - text_size[1] - 4),
            (text_x + text_size[0] + 8, text_y + 4),
            (0, 0, 0),
            -1,
        )
        self._draw_text_with_shadow(
            frame,
            text,
            (text_x, text_y),
            font_scale=0.35,
            color=ball_color,
            thickness=1,
        )

    def _draw_players(self, frame, players):
        """Draw all player objects with enhanced visuals."""
        for player in players:
            bbox = player.bbox.as_list()
            team = getattr(player, "team_id", None)
            if team == 1:
                color = self.visual_config.TEAM_1_COLOR
                team_name = "Team A"
            elif team == 0:
                color = self.visual_config.TEAM_2_COLOR
                team_name = "Team B"
            else:
                color = (128, 128, 128)
                team_name = "Unknown"
            y2 = int(bbox[3])
            x_center, _ = self._calculate_bbox_center(bbox)
            x_center = int(x_center)
            width = int(bbox[2] - bbox[0])
            ellipse_color = color
            shadow_color = tuple(c // 3 for c in color)
            cv2.ellipse(
                frame,
                center=(x_center + 2, y2 + 2),
                axes=(int(width * 0.6), int(0.35 * width)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=shadow_color,
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            cv2.ellipse(
                frame,
                center=(x_center, y2),
                axes=(int(width * 0.6), int(0.35 * width)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=ellipse_color,
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            track_id = getattr(player, "track_id", None)
            if track_id is not None:
                circle_radius = 12
                circle_x = x_center
                circle_y = y2 + 20
                cv2.circle(frame, (circle_x, circle_y), circle_radius, color, -1)
                cv2.circle(
                    frame, (circle_x, circle_y), circle_radius, (255, 255, 255), 1
                )
                text = f"{track_id}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                text_x = circle_x - text_size[0] // 2
                text_y = circle_y + text_size[1] // 2
                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            self._draw_minimal_player_speed(frame, player, x_center, int(bbox[1]) - 10)

    def _draw_minimal_player_speed(self, frame, player, x_center, y_position):
        """
        Draw minimal speed information - just text without background boxes.
        """
        speed = getattr(player, "speed", None)
        if speed is not None:
            speed_text = f"{speed:.0f}"
            cv2.putText(
                frame,
                speed_text,
                (x_center - 10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_ball_control(self, frame, ball_control_data: Dict[str, Any]):
        """
        Draws ball control information with modern, beautiful styling.
        Args:
            frame (np.ndarray): The frame to draw on.
            ball_control_data (dict): Ball control data containing percentages and counts.
        """
        frame_height, frame_width = frame.shape[:2]

        # Check if we have meaningful ball control data
        percentages = ball_control_data.get("percentages", {})
        total_frames = ball_control_data.get("total_frames_analyzed", 0)

        # Check frame-level ball control info
        controlling_player = ball_control_data.get("controlling_player")
        possession_team = ball_control_data.get("possession_team")
        control_confidence = ball_control_data.get("control_confidence")

        # If no meaningful data, show basic analysis panel
        if not percentages and controlling_player is None and possession_team is None:
            self._draw_no_data_indicator(frame)
            return

        # Enhanced modern overlay design (more compact)
        panel_width = 640
        panel_height = 120
        panel_x = 20
        panel_y = 20

        # Create semi-transparent background with gradient
        overlay = frame.copy()

        # Main panel background
        self._draw_rounded_rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            self.visual_config.BACKGROUND_COLOR,
            -1,
        )

        # Add gradient effect
        self._create_gradient_overlay(
            overlay,
            (40, 40, 40),  # Dark gray
            (20, 20, 20),  # Darker gray
            (panel_x, panel_y),
            (panel_width, panel_height),
        )

        # Blend overlay with frame
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Add border with accent color
        self._draw_rounded_rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            self.visual_config.ACCENT_COLOR,
            2,
        )

        # Title with refined typography
        title_y = panel_y + 35
        self._draw_text_with_shadow(
            frame,
            "BALL POSSESSION",
            (panel_x + 20, title_y),
            font_scale=0.6,
            color=self.visual_config.TEXT_COLOR,
            thickness=1,
        )

        # Draw possession information based on available data
        content_y = title_y + 30

        if percentages:
            # Draw possession bars if we have percentage data
            bar_height = 20
            bar_width = panel_width - 60

            for i, (team_id, percentage) in enumerate(percentages.items()):
                y_pos = content_y + (i * 40)

                # Team label
                team_color = (
                    self.visual_config.TEAM_1_COLOR
                    if team_id == "1"
                    else self.visual_config.TEAM_2_COLOR
                )
                team_name = f"Team {'A' if team_id == '1' else 'B'}"

                self._draw_text_with_shadow(
                    frame,
                    team_name,
                    (panel_x + 20, y_pos + 15),
                    font_scale=0.4,
                    color=team_color,
                    thickness=1,
                )

                # Possession bar background
                bar_x = panel_x + 100
                self._draw_rounded_rectangle(
                    frame,
                    (bar_x, y_pos),
                    (bar_x + bar_width, y_pos + bar_height),
                    (60, 60, 60),
                    -1,
                )

                # Possession bar fill
                fill_width = int(bar_width * percentage / 100)
                if fill_width > 0:
                    self._draw_rounded_rectangle(
                        frame,
                        (bar_x, y_pos),
                        (bar_x + fill_width, y_pos + bar_height),
                        team_color,
                        -1,
                    )

                # Percentage text
                percentage_text = f"{percentage:.1f}%"
                text_size = cv2.getTextSize(
                    percentage_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )[0]
                text_x = bar_x + bar_width + 10

                self._draw_text_with_shadow(
                    frame,
                    percentage_text,
                    (text_x, y_pos + 15),
                    font_scale=0.4,
                    color=self.visual_config.TEXT_COLOR,
                    thickness=1,
                )
        elif possession_team is not None:
            # Show current possession team if available
            team_color = (
                self.visual_config.TEAM_1_COLOR
                if possession_team == 1
                else self.visual_config.TEAM_2_COLOR
            )
            team_name = f"Team {'A' if possession_team == 1 else 'B'}"

            self._draw_text_with_shadow(
                frame,
                f"Current possession: {team_name}",
                (panel_x + 20, content_y + 15),
                font_scale=0.45,
                color=team_color,
                thickness=1,
            )

            if controlling_player is not None:
                self._draw_text_with_shadow(
                    frame,
                    f"Controlled by player: {controlling_player}",
                    (panel_x + 20, content_y + 45),
                    font_scale=0.35,
                    color=(200, 200, 200),
                    thickness=1,
                )

            if control_confidence is not None:
                confidence_text = f"Confidence: {control_confidence:.1f}%"
                self._draw_text_with_shadow(
                    frame,
                    confidence_text,
                    (panel_x + 20, content_y + 75),
                    font_scale=0.35,
                    color=(200, 200, 200),
                    thickness=1,
                )
        else:
            # Show basic analyzing message
            self._draw_text_with_shadow(
                frame,
                "Analyzing ball possession...",
                (panel_x + 20, content_y + 15),
                font_scale=0.4,
                color=(150, 150, 150),
                thickness=1,
            )

        # Statistics footer
        stats_y = panel_y + panel_height - 25
        if total_frames > 0:
            stats_text = f"Analyzed frames: {total_frames}"
        else:
            stats_text = "Real-time analysis"
        self._draw_text_with_shadow(
            frame,
            stats_text,
            (panel_x + 20, stats_y),
            font_scale=0.35,
            color=(180, 180, 180),
            thickness=1,
        )

    def _draw_no_data_indicator(self, frame):
        """Draw a modern indicator when no ball control data is available."""
        panel_width = 350
        panel_height = 100
        panel_x = 20
        panel_y = 20

        # Semi-transparent background
        overlay = frame.copy()
        self._draw_rounded_rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (40, 40, 40),
            -1,
        )
        cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)

        # Border
        self._draw_rounded_rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (100, 200, 255),
            2,
        )

        # Main text
        self._draw_text_with_shadow(
            frame,
            "FOOTBALL ANALYSIS - ENHANCED RENDERER",
            (panel_x + 20, panel_y + 35),
            font_scale=0.5,
            color=(100, 200, 255),
            thickness=1,
        )

        # Status text
        self._draw_text_with_shadow(
            frame,
            "Analyzing ball possession...",
            (panel_x + 20, panel_y + 65),
            font_scale=0.35,
            color=(200, 200, 200),
            thickness=1,
        )

    def _draw_no_detections_message(self, frame):
        """Draw a message when no detections are found."""
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        center_y = frame_height // 2
        message = "NO DETECTIONS FOUND"
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y
        padding = 20
        bg_x1 = text_x - padding
        bg_y1 = text_y - text_size[1] - padding
        bg_x2 = text_x + text_size[0] + padding
        bg_y2 = text_y + padding
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 0), 3)
        self._draw_text_with_shadow(
            frame,
            message,
            (text_x, text_y),
            font_scale=0.8,
            color=(255, 255, 0),
            thickness=1,
        )
        info_text = "Check if object detection is working properly"
        info_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        info_x = center_x - info_size[0] // 2
        info_y = center_y + 40
        self._draw_text_with_shadow(
            frame,
            info_text,
            (info_x, info_y),
            font_scale=0.5,
            color=(255, 255, 255),
            thickness=1,
        )

    def _calculate_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (center_x, center_y)
