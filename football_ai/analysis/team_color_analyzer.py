"""
Team Color Analysis Module

This module provides functionality to analyze team colors from player detections
and assign players to teams based on their jersey colors.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from collections import Counter

from ..domain.models import Detection, PlayerState, TeamColor
from ..domain.interfaces import TeamColorAnalyzer


class ModernTeamColorAnalyzer(TeamColorAnalyzer):
    """
    Modern implementation of team color analysis using K-means clustering
    and color histogram analysis to determine team assignments.
    """

    def __init__(
        self, n_clusters: int = 2, color_samples: int = 100, min_confidence: float = 0.6
    ):
        """
        Initialize the team color analyzer.

        Args:
            n_clusters: Number of color clusters (teams) to identify
            color_samples: Number of color samples to extract per player
            min_confidence: Minimum confidence threshold for team assignment
        """
        self.n_clusters = n_clusters
        self.color_samples = color_samples
        self.min_confidence = min_confidence
        self.team_colors: Optional[Dict[int, TeamColor]] = None
        self._color_history: Dict[int, List[np.ndarray]] = {}

    def analyze_frame_colors(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> Dict[int, TeamColor]:
        """
        Analyze team colors from player detections in a frame.

        Args:
            frame: Video frame as numpy array
            detections: List of player detections

        Returns:
            Dictionary mapping team IDs to team colors
        """
        player_colors = self._extract_player_colors(frame, detections)

        if len(player_colors) < 2:
            # Not enough players to determine teams
            return {}

        # Perform K-means clustering on dominant colors
        colors_array = np.array(list(player_colors.values()))
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        team_labels = kmeans.fit_predict(colors_array)

        # Determine team colors from cluster centers
        team_colors = {}
        for team_id in range(self.n_clusters):
            team_mask = team_labels == team_id
            if np.any(team_mask):
                team_color_bgr = kmeans.cluster_centers_[team_id].astype(int)
                team_colors[team_id] = TeamColor(
                    id=team_id,
                    primary_color=tuple(team_color_bgr),
                    name=f"Team_{team_id}",
                )

        self.team_colors = team_colors
        return team_colors

    def assign_player_team(
        self, frame: np.ndarray, detection: Detection
    ) -> Optional[int]:
        """
        Assign a player to a team based on their jersey color.

        Args:
            frame: Video frame as numpy array
            detection: Player detection

        Returns:
            Team ID or None if assignment is uncertain
        """
        if self.team_colors is None:
            return None

        player_color = self._extract_single_player_color(frame, detection)
        if player_color is None:
            return None

        # Find closest team color
        min_distance = float("inf")
        assigned_team = None

        for team_id, team_color in self.team_colors.items():
            distance = self._calculate_color_distance(
                player_color, np.array(team_color.primary_color)
            )

            if distance < min_distance:
                min_distance = distance
                assigned_team = team_id

        # Apply confidence threshold
        if min_distance > (1.0 - self.min_confidence) * 255 * 3:
            return None

        # Update color history for stability
        if detection.track_id is not None:
            if detection.track_id not in self._color_history:
                self._color_history[detection.track_id] = []

            self._color_history[detection.track_id].append(player_color)

            # Keep only recent history
            if len(self._color_history[detection.track_id]) > 10:
                self._color_history[detection.track_id] = self._color_history[
                    detection.track_id
                ][-10:]

            # Use majority vote from history
            team_votes = []
            for historical_color in self._color_history[detection.track_id]:
                vote = self._get_closest_team(historical_color)
                if vote is not None:
                    team_votes.append(vote)

            if team_votes:
                assigned_team = Counter(team_votes).most_common(1)[0][0]

        return assigned_team

    def _extract_player_colors(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> Dict[int, np.ndarray]:
        """Extract dominant colors for each player detection."""
        player_colors = {}

        for i, detection in enumerate(detections):
            if detection.class_name != "player":
                continue

            color = self._extract_single_player_color(frame, detection)
            if color is not None:
                player_colors[i] = color

        return player_colors

    def _extract_single_player_color(
        self, frame: np.ndarray, detection: Detection
    ) -> Optional[np.ndarray]:
        """Extract dominant color from a single player detection."""
        try:
            # Extract player region from frame
            x1, y1, x2, y2 = detection.bbox.to_xyxy()

            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(x1 + 1, min(int(x2), w))
            y2 = max(y1 + 1, min(int(y2), h))

            player_region = frame[y1:y2, x1:x2]

            if player_region.size == 0:
                return None

            # Focus on the upper part of the player (jersey area)
            jersey_height = int(player_region.shape[0] * 0.6)
            jersey_region = player_region[:jersey_height, :]

            if jersey_region.size == 0:
                return None

            # Reshape for color analysis
            pixels = jersey_region.reshape(-1, 3)

            # Remove very dark and very bright pixels (likely shadows/highlights)
            pixel_intensities = np.mean(pixels, axis=1)
            valid_mask = (pixel_intensities > 30) & (pixel_intensities < 220)

            if not np.any(valid_mask):
                # Fallback to all pixels if filtering removes everything
                valid_pixels = pixels
            else:
                valid_pixels = pixels[valid_mask]

            # Sample random pixels to reduce computation
            if len(valid_pixels) > self.color_samples:
                indices = np.random.choice(
                    len(valid_pixels), self.color_samples, replace=False
                )
                valid_pixels = valid_pixels[indices]

            # Find dominant color using K-means
            if len(valid_pixels) < 3:
                return np.mean(valid_pixels, axis=0)

            kmeans = KMeans(
                n_clusters=min(3, len(valid_pixels)), random_state=42, n_init=10
            )
            kmeans.fit(valid_pixels)

            # Return the most frequent cluster center
            labels = kmeans.labels_
            label_counts = Counter(labels)
            dominant_label = label_counts.most_common(1)[0][0]
            dominant_color = kmeans.cluster_centers_[dominant_label]

            return dominant_color

        except Exception as e:
            print(f"Error extracting player color: {e}")
            return None

    def _calculate_color_distance(
        self, color1: np.ndarray, color2: np.ndarray
    ) -> float:
        """Calculate Euclidean distance between two colors."""
        return float(np.linalg.norm(color1 - color2))

    def _get_closest_team(self, player_color: np.ndarray) -> Optional[int]:
        """Get the closest team for a given player color."""
        if self.team_colors is None:
            return None

        min_distance = float("inf")
        closest_team = None

        for team_id, team_color in self.team_colors.items():
            distance = self._calculate_color_distance(
                player_color, np.array(team_color.primary_color)
            )

            if distance < min_distance:
                min_distance = distance
                closest_team = team_id

        return closest_team

    def get_team_colors(self) -> Optional[Dict[int, TeamColor]]:
        """Get the current team colors."""
        return self.team_colors

    def reset(self):
        """Reset the analyzer state."""
        self.team_colors = None
        self._color_history.clear()
