"""
Team Feature Analysis Module

Analyzes video frames to identify team distinguishing features using various techniques.
Currently focuses on color analysis using K-means clustering, but designed to be
extensible for other features like patterns, logos, etc.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
from sklearn.cluster import KMeans

from ..domain.models import Detection, TeamColor, TeamFeatures
from ..domain.interfaces import TeamFeatureAnalyzer


class KMeansTeamColorAnalyzer(TeamFeatureAnalyzer):
    """
    Analyzes video frames to identify team colors using K-means clustering.
    Implements color-based team feature analysis.
    """

    def __init__(self, min_players: int = 6, min_separation: float = 50.0):
        """
        Initialize the team feature analyzer.

        Args:
            min_players: Minimum number of players needed for reliable analysis
            min_separation: Minimum color distance between teams for reliable detection
        """
        self.min_players = min_players
        self.min_separation = min_separation

    def analyze_team_features(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> Optional[Dict[int, TeamFeatures]]:
        """
        Analyze team colors from player detections.

        Args:
            frame: Video frame
            detections: List of player detections

        Returns:
            Dictionary mapping team IDs to team colors, or None if analysis fails
        """
        if len(detections) < self.min_players:
            return None

        # Extract colors from all players
        player_colors = []

        for detection in detections:
            color = self._extract_player_color(frame, detection)
            if color is not None:
                # Filter out very dark colors (likely shadows/occlusions)
                brightness = np.mean(color)
                if brightness > 30:  # Minimum brightness threshold
                    player_colors.append(color)

        if len(player_colors) < self.min_players:
            return None

        # Simple K-means clustering into 2 teams with better initialization
        try:
            player_colors_array = np.array(player_colors)

            # Try multiple K-means runs and pick the best one
            best_kmeans = None
            best_inertia = float("inf")

            for _ in range(5):  # Multiple attempts for better clustering
                kmeans = KMeans(
                    n_clusters=2, random_state=None, n_init=10, max_iter=300
                )
                kmeans.fit(player_colors_array)

                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_kmeans = kmeans

            if best_kmeans is None:
                return None

            # Validate that we have meaningful separation between teams
            team_centers = best_kmeans.cluster_centers_
            team_separation = np.linalg.norm(team_centers[0] - team_centers[1])

            if team_separation < self.min_separation:
                print(
                    f"Teams not visually distinct enough (separation: {team_separation:.1f})"
                )
                return None

            # Create team colors
            team_colors = {}
            for team_id in range(2):
                team_color_bgr = best_kmeans.cluster_centers_[team_id].astype(int)
                team_colors[team_id] = TeamColor(
                    id=team_id,
                    primary_color=tuple(team_color_bgr),
                    name=f"Team_{team_id}",
                )

            self.team_colors = team_colors

            # Print debug info about team colors detected
            print(f"Team colors detected:")
            print(f"  Team 0: BGR{team_colors[0].primary_color}")
            print(f"  Team 1: BGR{team_colors[1].primary_color}")
            print(f"  Separation distance: {team_separation:.1f}")

            return team_colors

        except Exception as e:
            print(f"Team color analysis failed: {e}")
            return None

    def _extract_player_color(
        self, frame: np.ndarray, detection: Detection
    ) -> Optional[np.ndarray]:
        """
        Extract dominant color from player's jersey area with improved robustness.
        Uses multiple sampling strategies and outlier removal.
        """
        try:
            bbox = detection.bbox
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

            # Ensure bbox is within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            # Focus on jersey area - avoid head and legs
            player_height = y2 - y1
            player_width = x2 - x1

            # Jersey region: skip top 15% (head) and bottom 40% (legs/shorts)
            jersey_y_start = y1 + int(player_height * 0.15)
            jersey_y_end = y1 + int(player_height * 0.6)

            # Narrow horizontally to avoid arms/background
            jersey_x_start = x1 + int(player_width * 0.2)
            jersey_x_end = x2 - int(player_width * 0.2)

            # Ensure valid region - fallback to simpler extraction if needed
            if jersey_y_end <= jersey_y_start or jersey_x_end <= jersey_x_start:
                # Simple fallback: upper 50% of bounding box
                jersey_height = int(player_height * 0.5)
                jersey_y_start = y1 + int(player_height * 0.1)
                jersey_roi = frame[
                    jersey_y_start : jersey_y_start + jersey_height, x1:x2
                ]
            else:
                jersey_roi = frame[
                    jersey_y_start:jersey_y_end, jersey_x_start:jersey_x_end
                ]

            if jersey_roi.size == 0:
                return None

            # Get pixels and remove outliers (helps with shadows, reflections)
            pixels = jersey_roi.reshape(-1, 3).astype(np.float32)

            # Remove very dark (shadows) and very bright (reflections) pixels
            brightness = np.mean(pixels, axis=1)
            brightness_threshold_low = np.percentile(brightness, 20)
            brightness_threshold_high = np.percentile(brightness, 80)

            valid_mask = (brightness >= brightness_threshold_low) & (
                brightness <= brightness_threshold_high
            )

            if (
                np.sum(valid_mask) < len(pixels) * 0.1
            ):  # If too few valid pixels, use all
                filtered_pixels = pixels
            else:
                filtered_pixels = pixels[valid_mask]

            # Get median color (more robust than mean)
            median_color = np.median(filtered_pixels, axis=0)
            return median_color

        except Exception:
            return None
