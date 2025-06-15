"""
Team Assignment Module

Handles assignment of players to teams based on established team colors.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from ..domain.models import Detection, TeamAssignment, TeamFeatures
from ..domain.interfaces import TeamAssigner as TeamAssignerInterface


class TeamAssigner(TeamAssignerInterface):
    """
    Handles assignment of players to teams based on established team features.
    Manages caching of team assignments by track ID.
    """

    def __init__(self, confidence_threshold: float = 80.0):
        """
        Initialize the team assigner.

        Args:
            confidence_threshold: Maximum feature distance for confident assignment
        """
        self._team_features: Optional[Dict[int, TeamFeatures]] = None
        self._team_assignments_by_track_id: Dict[int, TeamAssignment] = {}
        self.confidence_threshold = confidence_threshold

    def set_team_features(self, team_features: Dict[int, TeamFeatures]) -> None:
        """Set the established team features for assignment."""
        if len(team_features) >= 2:
            self._team_features = team_features
        else:
            raise ValueError("Need at least 2 team feature sets for assignment")

    def has_team_features(self) -> bool:
        """Check if team features have been established."""
        return self._team_features is not None

    def get_player_team_assignment(self, track_id: int) -> Optional[TeamAssignment]:
        """Get cached team assignment for a player by track ID."""
        return self._team_assignments_by_track_id.get(track_id)

    def assign_player_team(
        self, frame: np.ndarray, detection: Detection
    ) -> Optional[TeamAssignment]:
        """
        Assign a player to a team based on their features.

        Args:
            frame: Video frame
            detection: Player detection

        Returns:
            TeamAssignment or None if assignment fails
        """
        if not self._team_features or detection.track_id is None:
            return None

        # Check if already assigned
        if detection.track_id in self._team_assignments_by_track_id:
            return self._team_assignments_by_track_id[detection.track_id]

        # Extract player features
        player_features = self._extract_player_features(frame, detection)
        if not player_features:
            return None

        # Find closest team features with confidence threshold
        min_distance = float("inf")
        assigned_team = None

        for team_id, team_features in self._team_features.items():
            distance = self._compute_feature_distance(player_features, team_features)
            if distance < min_distance:
                min_distance = distance
                assigned_team = team_id

        # Only assign if we're confident enough
        if min_distance > self.confidence_threshold:
            assignment = TeamAssignment.UNKNOWN
        elif assigned_team == 0:
            assignment = TeamAssignment.TEAM_1
        elif assigned_team == 1:
            assignment = TeamAssignment.TEAM_2
        else:
            assignment = TeamAssignment.UNKNOWN

        # Cache the assignment
        self._team_assignments_by_track_id[detection.track_id] = assignment
        return assignment

    def assign_players_batch(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> Dict[int, TeamAssignment]:
        """
        Assign multiple players to teams in batch.

        Args:
            frame: Video frame
            detections: List of player detections

        Returns:
            Dictionary mapping track_id to TeamAssignment
        """
        assignments = {}
        for detection in detections:
            if detection.track_id is not None:
                assignment = self.assign_player_team(frame, detection)
                if assignment:
                    assignments[detection.track_id] = assignment
        return assignments

    def get_all_assignments(self) -> Dict[int, TeamAssignment]:
        """Get all cached team assignments."""
        return self._team_assignments_by_track_id.copy()

    def clear_assignments(self) -> None:
        """Clear all cached assignments."""
        self._team_assignments_by_track_id.clear()

    def get_assignment_stats(self) -> Dict[str, int]:
        """Get statistics about current assignments."""
        stats = {
            "total_assigned": len(self._team_assignments_by_track_id),
            "team_1": sum(
                1
                for a in self._team_assignments_by_track_id.values()
                if a == TeamAssignment.TEAM_1
            ),
            "team_2": sum(
                1
                for a in self._team_assignments_by_track_id.values()
                if a == TeamAssignment.TEAM_2
            ),
            "unknown": sum(
                1
                for a in self._team_assignments_by_track_id.values()
                if a == TeamAssignment.UNKNOWN
            ),
        }
        return stats

    def _extract_player_color(
        self, frame: np.ndarray, detection: Detection
    ) -> Optional[np.ndarray]:
        """
        Extract dominant color from player's jersey area.

        Args:
            frame: Video frame
            detection: Player detection

        Returns:
            BGR color array or None if extraction fails
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

    def _extract_player_features(
        self, frame: np.ndarray, detection: Detection
    ) -> Optional[Dict[str, Any]]:
        """Extract features from a player for team assignment."""
        features = {}

        # For now, extract color features (can be extended for other features)
        color = self._extract_player_color(frame, detection)
        if color is not None:
            features["color"] = color

        return features if features else None

    def _compute_feature_distance(
        self, player_features: Dict[str, Any], team_features: TeamFeatures
    ) -> float:
        """Compute distance between player features and team features."""
        total_distance = 0.0
        feature_count = 0

        # Color distance
        if "color" in player_features and team_features.has_feature("color"):
            player_color = player_features["color"]
            team_color = team_features.get_feature("color")
            if team_color is not None:
                color_distance = np.linalg.norm(
                    np.array(player_color) - np.array(team_color)
                )
                total_distance += color_distance
                feature_count += 1

        # Add other feature distances here as needed
        # if 'pattern' in player_features and team_features.has_feature('pattern'):
        #     ...

        return (
            float(total_distance / feature_count) if feature_count > 0 else float("inf")
        )
