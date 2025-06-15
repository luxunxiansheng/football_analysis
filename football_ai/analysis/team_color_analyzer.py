"""
Team Color Analysis Module

Simple team color analysis for distinguishing between two teams.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
from sklearn.cluster import KMeans

from ..domain.models import Detection, TeamColor
from ..domain.interfaces import TeamColorAnalyzer


class KMeansTeamColorAnalyzer(TeamColorAnalyzer):
    """
    Simple team color analyzer that extracts player jersey colors
    and assigns them to teams using basic clustering.
    """

    def __init__(self):
        """Initialize the team color analyzer."""
        self.team_colors: Optional[Dict[int, TeamColor]] = None

    def analyze_frame_colors(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> Dict[int, TeamColor]:
        """
        Analyze team colors from player detections.

        Args:
            frame: Video frame
            detections: List of player detections

        Returns:
            Dictionary mapping team IDs to team colors
        """
        if len(detections) < 4:
            return {}

        # Extract colors from all players
        player_colors = []
        for detection in detections:
            color = self._extract_player_color(frame, detection)
            if color is not None:
                player_colors.append(color)

        if len(player_colors) < 4:
            return {}

        # Simple K-means clustering into 2 teams
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(player_colors)

            # Create team colors
            team_colors = {}
            for team_id in range(2):
                team_color_bgr = kmeans.cluster_centers_[team_id].astype(int)
                team_colors[team_id] = TeamColor(
                    id=team_id,
                    primary_color=tuple(team_color_bgr),
                    name=f"Team_{team_id}",
                )

            self.team_colors = team_colors
            return team_colors

        except Exception:
            return {}

    def assign_player_team(
        self, frame: np.ndarray, detection: Detection
    ) -> Optional[int]:
        """
        Assign a player to a team based on jersey color.

        Args:
            frame: Video frame
            detection: Player detection

        Returns:
            Team ID (0 or 1) or None
        """
        if self.team_colors is None:
            return None

        player_color = self._extract_player_color(frame, detection)
        if player_color is None:
            return None

        # Find closest team color
        min_distance = float("inf")
        assigned_team = None

        for team_id, team_color in self.team_colors.items():
            distance = np.linalg.norm(player_color - np.array(team_color.primary_color))
            if distance < min_distance:
                min_distance = distance
                assigned_team = team_id

        return assigned_team

    def assign_player_team_preliminary(
        self, frame: np.ndarray, detection: Detection
    ) -> Optional[int]:
        """
        Preliminary team assignment before full analysis.
        Uses simple color characteristics.

        Args:
            frame: Video frame
            detection: Player detection

        Returns:
            Team ID (0 or 1) or None
        """
        player_color = self._extract_player_color(frame, detection)
        if player_color is None:
            return None

        # Simple assignment based on color brightness and hue
        # This is just a placeholder until full analysis is done
        brightness = np.mean(player_color)

        # Convert BGR to HSV for hue analysis
        try:
            bgr_pixel = player_color.reshape(1, 1, 3).astype(np.uint8)
            hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
            hue = float(hsv_pixel[0, 0, 0])

            # Simple hue-based assignment
            if hue < 60 or hue > 120:  # Red/Blue spectrum
                return 0
            else:  # Green/Yellow spectrum
                return 1

        except Exception:
            # Fallback to brightness-based assignment
            return 0 if brightness > 127 else 1

    def _extract_player_color(
        self, frame: np.ndarray, detection: Detection
    ) -> Optional[np.ndarray]:
        """Extract dominant color from player's jersey area."""
        try:
            bbox = detection.bbox
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

            # Ensure bbox is within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            # Extract upper portion (jersey area)
            jersey_height = int((y2 - y1) * 0.6)
            jersey_roi = frame[y1 : y1 + jersey_height, x1:x2]

            if jersey_roi.size == 0:
                return None

            # Get mean color
            mean_color = np.mean(jersey_roi.reshape(-1, 3), axis=0)
            return mean_color

        except Exception:
            return None

    def get_team_colors(self) -> Optional[Dict[int, TeamColor]]:
        """Get current team colors."""
        return self.team_colors

    def reset(self):
        """Reset analyzer state."""
        self.team_colors = None
