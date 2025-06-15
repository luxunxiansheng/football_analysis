"""
Video Rendering Module

This module provides functionality to render analysis results onto video frames
with annotations, tracks, and statistics visualization.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

from ..domain.models import Detection, FieldEntityState, TeamAssignment, TeamColor


class VideoRenderer:
    """
    Implementation of video rendering with clean annotations
    and professional visualization of football analysis results.
    """

    def __init__(
        self,
        show_tracks: bool = True,
        show_team_colors: bool = True,
        show_ball_possession: bool = True,
        show_speeds: bool = True,
        show_positions: bool = True,
    ):
        """
        Initialize the video renderer.

        Args:
            show_tracks: Whether to show player tracking lines
            show_team_colors: Whether to show team color indicators
            show_ball_possession: Whether to show ball possession indicators
            show_speeds: Whether to show player speeds
            show_positions: Whether to show transformed positions
        """
        self.show_tracks = show_tracks
        self.show_team_colors = show_team_colors
        self.show_ball_possession = show_ball_possession
        self.show_speeds = show_speeds
        self.show_positions = show_positions

        # Color scheme
        self.colors = {
            "team_1": (0, 255, 0),  # Green
            "team_2": (0, 0, 255),  # Blue
            "referee": (255, 255, 0),  # Yellow
            "ball": (255, 255, 255),  # White
            "unknown": (128, 128, 128),  # Gray
            "possession": (255, 0, 255),  # Magenta
            "track": (200, 200, 200),  # Light gray
            "text": (255, 255, 255),  # White
            "background": (0, 0, 0),  # Black
        }

        # Track history for visualization
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.max_track_length = 30

    def render_frame(
        self,
        frame: np.ndarray,
        player_states: List[FieldEntityState],
        ball_detections: List[Detection],
        referee_detections: List[Detection],
        team_colors: Optional[Dict[int, TeamColor]] = None,
        possession_info: Optional[Dict[str, Any]] = None,
        camera_movement: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Render all analysis results onto a video frame.

        Args:
            frame: Original video frame
            player_states: List of player states with tracking info
            ball_detections: List of ball detections
            referee_detections: List of referee detections
            team_colors: Team color information
            possession_info: Ball possession analysis results
            camera_movement: Camera movement for current frame

        Returns:
            Annotated frame with all visualizations
        """
        # Create a copy to avoid modifying the original
        annotated_frame = frame.copy()

        # Draw player tracks first (background layer)
        if self.show_tracks:
            annotated_frame = self._draw_tracks(annotated_frame)

        # Draw referee detections
        for referee in referee_detections:
            annotated_frame = self._draw_referee(annotated_frame, referee)

        # Draw ball detections
        for ball in ball_detections:
            annotated_frame = self._draw_ball(annotated_frame, ball, possession_info)

        # Draw player states
        for player in player_states:
            annotated_frame = self._draw_player(
                annotated_frame, player, team_colors, possession_info
            )

            # Update track history
            if player.track_id is not None:
                center = (int(player.bbox.center_x), int(player.bbox.center_y))
                if player.track_id not in self.track_history:
                    self.track_history[player.track_id] = []

                self.track_history[player.track_id].append(center)

                # Limit track length
                if len(self.track_history[player.track_id]) > self.max_track_length:
                    self.track_history[player.track_id] = self.track_history[
                        player.track_id
                    ][-self.max_track_length :]

        # Draw overlay information
        annotated_frame = self._draw_overlay(
            annotated_frame, team_colors, possession_info, camera_movement
        )

        return annotated_frame

    def _draw_player(
        self,
        frame: np.ndarray,
        player: FieldEntityState,
        team_colors: Optional[Dict[int, TeamColor]] = None,
        possession_info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Draw a single player with all annotations."""
        bbox = player.bbox
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

        # Determine player color
        if player.team == TeamAssignment.TEAM_1:
            color = self.colors["team_1"]
        elif player.team == TeamAssignment.TEAM_2:
            color = self.colors["team_2"]
        else:
            color = self.colors["unknown"]

        # Use team color if available and enabled
        if (
            self.show_team_colors
            and player.team_color is not None
            and len(player.team_color) >= 3
        ):
            # Convert BGR to RGB for display and ensure integers
            color = (
                int(player.team_color[2]),
                int(player.team_color[1]),
                int(player.team_color[0]),
            )

        # Draw bounding box with different styles for preliminary vs confirmed
        if (
            hasattr(player, "team_assignment_confidence")
            and player.team_assignment_confidence == "preliminary"
        ):
            # Dashed line for preliminary assignments
            thickness = 3 if player.has_ball else 2
            self._draw_dashed_rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Add small "P" indicator for preliminary
            cv2.putText(
                frame,
                "P",
                (x2 - 15, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
        else:
            # Solid line for confirmed assignments
            thickness = 3 if player.has_ball else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw possession indicator
        if self.show_ball_possession and player.has_ball:
            cv2.circle(
                frame, (int(bbox.center_x), y1 - 10), 8, self.colors["possession"], -1
            )
            cv2.putText(
                frame,
                "BALL",
                (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors["possession"],
                2,
            )

        # Draw track ID
        if player.track_id is not None:
            cv2.putText(
                frame,
                f"ID:{player.track_id}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # Draw speed information
        if self.show_speeds and player.speed is not None:
            speed_text = f"{player.speed:.1f} km/h"
            cv2.putText(
                frame,
                speed_text,
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # Draw transformed position
        if self.show_positions and player.position_transformed is not None:
            pos_text = f"({player.position_transformed[0]:.1f}, {player.position_transformed[1]:.1f})"
            cv2.putText(
                frame, pos_text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1
            )

        return frame

    def _draw_ball(
        self,
        frame: np.ndarray,
        ball: Detection,
        possession_info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Draw ball detection with possession information."""
        bbox = ball.bbox
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        center = (int(bbox.center_x), int(bbox.center_y))

        # Draw ball bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["ball"], 2)

        # Draw ball center
        cv2.circle(frame, center, 5, self.colors["ball"], -1)

        # Draw confidence
        conf_text = f"{ball.confidence:.2f}"
        cv2.putText(
            frame,
            conf_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.colors["text"],
            1,
        )

        # Draw possession line to controlling player
        if (
            possession_info
            and possession_info.get("possessor_id") is not None
            and possession_info.get("min_distance", float("inf")) < 100
        ):

            # Draw line from ball to closest player (would need player position)
            pass

        return frame

    def _draw_referee(self, frame: np.ndarray, referee: Detection) -> np.ndarray:
        """Draw referee detection."""
        bbox = referee.bbox
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

        # Draw referee bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["referee"], 2)

        # Label
        cv2.putText(
            frame,
            "REF",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.colors["referee"],
            1,
        )

        # Draw track ID if available
        if referee.track_id is not None:
            cv2.putText(
                frame,
                f"ID:{referee.track_id}",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.colors["referee"],
                1,
            )

        return frame

    def _draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Draw player movement tracks."""
        for track_id, track_points in self.track_history.items():
            if len(track_points) < 2:
                continue

            # Draw track line
            points = np.array(track_points, dtype=np.int32)
            cv2.polylines(frame, [points], False, self.colors["track"], 1)

            # Draw direction arrow at the end
            if len(track_points) >= 2:
                start_point = track_points[-2]
                end_point = track_points[-1]

                # Calculate arrow direction
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]

                if dx != 0 or dy != 0:  # Avoid division by zero
                    length = np.sqrt(dx * dx + dy * dy)
                    if length > 5:  # Only draw arrow if movement is significant
                        # Normalize and scale
                        dx = int(dx / length * 10)
                        dy = int(dy / length * 10)

                        # Draw arrow
                        cv2.arrowedLine(
                            frame,
                            (end_point[0] - dx, end_point[1] - dy),
                            end_point,
                            self.colors["track"],
                            2,
                            tipLength=0.3,
                        )

        return frame

    def _draw_overlay(
        self,
        frame: np.ndarray,
        team_colors: Optional[Dict[int, TeamColor]] = None,
        possession_info: Optional[Dict[str, Any]] = None,
        camera_movement: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Draw overlay information like statistics and legends."""
        h, w = frame.shape[:2]

        # Draw semi-transparent overlay panel
        overlay = frame.copy()
        panel_height = 120
        cv2.rectangle(
            overlay, (10, 10), (400, panel_height), self.colors["background"], -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y_pos = 30

        # Team colors legend
        if team_colors and self.show_team_colors:
            cv2.putText(
                frame,
                "Team Colors:",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors["text"],
                1,
            )
            y_pos += 20

            for team_id, team_color in team_colors.items():
                # Safely extract color with validation
                if (
                    hasattr(team_color, "primary_color")
                    and team_color.primary_color is not None
                    and len(team_color.primary_color) >= 3
                ):
                    # Convert BGR to RGB and ensure values are integers
                    color = (
                        int(team_color.primary_color[2]),
                        int(team_color.primary_color[1]),
                        int(team_color.primary_color[0]),
                    )
                else:
                    # Fallback colors for teams
                    fallback_colors = [
                        (255, 0, 0),
                        (0, 0, 255),
                        (0, 255, 0),
                        (255, 255, 0),
                    ]
                    color = fallback_colors[team_id % len(fallback_colors)]

                cv2.rectangle(frame, (20, y_pos - 10), (35, y_pos + 5), color, -1)

                # Safely get team name
                team_name = getattr(team_color, "name", f"Team_{team_id}")
                cv2.putText(
                    frame,
                    team_name,
                    (45, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    self.colors["text"],
                    1,
                )
                y_pos += 20

        # Ball possession info
        if possession_info and self.show_ball_possession:
            possessor = possession_info.get("possessor_id")
            team_possession = possession_info.get("team_possession")

            if possessor is not None:
                cv2.putText(
                    frame,
                    f"Ball Control: Player {possessor}",
                    (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    self.colors["possession"],
                    1,
                )
                y_pos += 15

            if team_possession is not None:
                team_name = (
                    f"Team {team_possession.value}"
                    if hasattr(team_possession, "value")
                    else str(team_possession)
                )
                cv2.putText(
                    frame,
                    f"Possession: {team_name}",
                    (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    self.colors["text"],
                    1,
                )
                y_pos += 15

        # Camera movement info
        if camera_movement and len(camera_movement) >= 2:
            movement_text = (
                f"Camera: ({camera_movement[0]:.1f}, {camera_movement[1]:.1f})"
            )
            cv2.putText(
                frame,
                movement_text,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.colors["text"],
                1,
            )

        return frame

    def reset_tracks(self):
        """Reset all track history."""
        self.track_history.clear()

    def set_colors(self, new_colors: Dict[str, Tuple[int, int, int]]):
        """Update color scheme."""
        self.colors.update(new_colors)

    def get_track_history(self) -> Dict[int, List[Tuple[int, int]]]:
        """Get current track history."""
        return self.track_history.copy()

    def _draw_dashed_rectangle(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        dash_length: int = 8,
    ) -> None:
        """Draw a dashed rectangle for preliminary team assignments."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Draw dashed lines for each side of the rectangle
        self._draw_dashed_line(
            frame, (x1, y1), (x2, y1), color, thickness, dash_length
        )  # Top
        self._draw_dashed_line(
            frame, (x2, y1), (x2, y2), color, thickness, dash_length
        )  # Right
        self._draw_dashed_line(
            frame, (x2, y2), (x1, y2), color, thickness, dash_length
        )  # Bottom
        self._draw_dashed_line(
            frame, (x1, y2), (x1, y1), color, thickness, dash_length
        )  # Left

    def _draw_dashed_line(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        dash_length: int = 8,
    ) -> None:
        """Draw a dashed line between two points."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Calculate line parameters
        dx = x2 - x1
        dy = y2 - y1
        line_length = np.sqrt(dx**2 + dy**2)

        if line_length == 0:
            return

        # Normalize direction
        unit_x = dx / line_length
        unit_y = dy / line_length

        # Draw dashed line
        current_length = 0
        draw_dash = True

        while current_length < line_length:
            next_length = min(current_length + dash_length, line_length)

            if draw_dash:
                start_x = int(x1 + current_length * unit_x)
                start_y = int(y1 + current_length * unit_y)
                end_x = int(x1 + next_length * unit_x)
                end_y = int(y1 + next_length * unit_y)

                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)

            current_length = next_length
            draw_dash = not draw_dash
