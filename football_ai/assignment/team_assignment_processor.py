"""
Improved Team Assignment Processor with temporal consistency and track awareness.
"""

import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

from ..domain.data_models import VideoData, FrameData
from ..domain.interfaces import Processor


class TeamFeatureExtractor:
    """Enhanced feature extractor that considers multiple aspects of player appearance."""

    def extract(self, frame, detection) -> Dict[str, Any]:
        """Extract comprehensive features for team classification."""
        x1, y1, x2, y2 = map(
            int,
            [
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2,
            ],
        )

        h, w = y2 - y1, x2 - x1

        # Extract jersey region (upper portion)
        jersey_y = y1 + h // 4  # Start from chest area
        jersey_crop = frame[jersey_y : y1 + h // 2, x1:x2]

        # Extract shorts region (lower portion)
        shorts_y = y1 + 2 * h // 3
        shorts_crop = frame[shorts_y:y2, x1:x2]

        features = {}

        # 1. Jersey color features
        if jersey_crop.size > 0:
            jersey_mean = jersey_crop.mean(axis=(0, 1))
            jersey_std = jersey_crop.std(axis=(0, 1))
            # Dominant color (most frequent color cluster)
            jersey_dominant = self._get_dominant_color(jersey_crop)
            features["jersey_mean"] = jersey_mean.tolist()
            features["jersey_std"] = jersey_std.tolist()
            features["jersey_dominant"] = jersey_dominant
        else:
            features["jersey_mean"] = [0, 0, 0]
            features["jersey_std"] = [0, 0, 0]
            features["jersey_dominant"] = [0, 0, 0]

        # 2. Shorts color features
        if shorts_crop.size > 0:
            shorts_mean = shorts_crop.mean(axis=(0, 1))
            features["shorts_mean"] = shorts_mean.tolist()
        else:
            features["shorts_mean"] = [0, 0, 0]

        # 3. Overall brightness and contrast
        if jersey_crop.size > 0:
            features["brightness"] = float(np.mean(jersey_crop))
            features["contrast"] = float(np.std(jersey_crop))
        else:
            features["brightness"] = 0.0
            features["contrast"] = 0.0

        return features

    def _get_dominant_color(self, crop, k=3):
        """Extract dominant color using simple color clustering."""
        if crop.size == 0:
            return [0, 0, 0]

        # Reshape to list of pixels
        pixels = crop.reshape(-1, 3)

        # Simple dominant color - just the median
        dominant = np.median(pixels, axis=0)
        return dominant.tolist()


class TrackAwareTeamAssigner:
    """Team assigner that maintains consistency across tracks and frames."""

    def __init__(self, n_teams=2, consistency_weight=0.7, min_confidence=0.6):
        self.n_teams = n_teams
        self.consistency_weight = consistency_weight
        self.min_confidence = min_confidence

        # Track assignment history: track_id -> [team assignments]
        self.track_history: Dict[int, List[int]] = defaultdict(list)

        # Global team characteristics learned over time
        self.team_profiles: Dict[int, Dict] = {}

        # Frame counter for temporal smoothing
        self.frame_count = 0

    def assign_teams(
        self, features_list: List[Dict], track_ids: List[int]
    ) -> List[int]:
        """Assign teams considering temporal consistency and track history."""

        if not features_list:
            return []

        self.frame_count += 1

        # Convert features to numerical arrays for clustering
        feature_vectors = self._features_to_vectors(features_list)

        # Get initial clustering assignments
        if self.frame_count <= 10:  # Bootstrap phase
            initial_assignments = self._bootstrap_assignment(feature_vectors)
        else:
            initial_assignments = self._consistent_assignment(
                feature_vectors, track_ids
            )

        # Apply temporal smoothing based on track history
        final_assignments = self._apply_temporal_smoothing(
            initial_assignments, track_ids, features_list
        )

        # Update track history
        for track_id, assignment in zip(track_ids, final_assignments):
            if track_id > 0:  # Valid track ID
                self.track_history[track_id].append(assignment)
                # Keep only recent history (last 30 frames)
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id] = self.track_history[track_id][-30:]

        # Update team profiles
        self._update_team_profiles(features_list, final_assignments)

        return final_assignments

    def _features_to_vectors(self, features_list: List[Dict]) -> np.ndarray:
        """Convert feature dictionaries to numerical vectors."""
        vectors = []
        for features in features_list:
            vector = []
            vector.extend(features["jersey_mean"])
            vector.extend(features["jersey_std"])
            vector.extend(features["shorts_mean"])
            vector.append(features["brightness"])
            vector.append(features["contrast"])
            vectors.append(vector)
        return np.array(vectors)

    def _bootstrap_assignment(self, feature_vectors: np.ndarray) -> List[int]:
        """Initial team assignment using K-means clustering."""
        if len(feature_vectors) < self.n_teams:
            return [0] * len(feature_vectors)

        # Use K-means for initial clustering
        from sklearn.cluster import KMeans

        try:
            kmeans = KMeans(n_clusters=self.n_teams, random_state=42, n_init=10)
            labels = kmeans.fit_predict(feature_vectors)
            return [(label + 1) for label in labels]  # Convert to 1-based team IDs
        except ImportError:
            # Fallback simple clustering if sklearn not available
            return self._simple_clustering(feature_vectors)

    def _simple_clustering(self, feature_vectors: np.ndarray) -> List[int]:
        """Simple K-means implementation as fallback."""
        n_samples = len(feature_vectors)
        if n_samples < self.n_teams:
            return [1] * n_samples

        # Initialize centroids
        np.random.seed(42)
        centroids = feature_vectors[
            np.random.choice(n_samples, self.n_teams, replace=False)
        ]  # K-means iterations
        for _ in range(10):
            distances = np.linalg.norm(
                feature_vectors[:, None] - centroids[None, :], axis=2
            )
            labels = distances.argmin(axis=1)

            for i in range(self.n_teams):
                mask = labels == i
                if np.any(mask):
                    centroids[i] = feature_vectors[mask].mean(axis=0)

        # Convert to 1-based team IDs and return as list
        return [(label + 1) for label in labels]

    def _consistent_assignment(
        self, feature_vectors: np.ndarray, track_ids: List[int]
    ) -> List[int]:
        """Assign teams maintaining consistency with established team profiles."""
        assignments = []

        for i, (features, track_id) in enumerate(zip(feature_vectors, track_ids)):
            if track_id > 0 and track_id in self.track_history:
                # Use track history for consistent assignment
                history = self.track_history[track_id]
                if len(history) >= 3:  # Enough history
                    most_common_team = Counter(history).most_common(1)[0][0]
                    assignments.append(most_common_team)
                    continue

            # Assign based on similarity to team profiles
            if self.team_profiles:
                team_similarities = []
                for team_id in range(1, self.n_teams + 1):
                    if team_id in self.team_profiles:
                        similarity = self._calculate_similarity(features, team_id)
                        team_similarities.append((team_id, similarity))

                if team_similarities:
                    best_team = max(team_similarities, key=lambda x: x[1])[0]
                    assignments.append(best_team)
                    continue

            # Fallback: assign to team 1
            assignments.append(1)

        return assignments

    def _calculate_similarity(self, features: np.ndarray, team_id: int) -> float:
        """Calculate similarity between features and team profile."""
        if team_id not in self.team_profiles:
            return 0.0

        profile = self.team_profiles[team_id]
        profile_vector = np.array(profile["mean_features"])

        # Euclidean distance similarity
        distance = np.linalg.norm(features - profile_vector)
        similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity

        return float(similarity)

    def _apply_temporal_smoothing(
        self, assignments: List[int], track_ids: List[int], features_list: List[Dict]
    ) -> List[int]:
        """Apply temporal smoothing to reduce assignment flickering."""
        smoothed = []

        for assignment, track_id, features in zip(
            assignments, track_ids, features_list
        ):
            if track_id > 0 and track_id in self.track_history:
                history = self.track_history[track_id][-10:]  # Last 10 assignments

                if len(history) >= 3:
                    # Get most common assignment in recent history
                    team_counts = Counter(history)
                    most_common_team, count = team_counts.most_common(1)[0]
                    confidence = count / len(history)

                    # Use historical assignment if confidence is high
                    if confidence >= self.min_confidence:
                        smoothed.append(most_common_team)
                    else:
                        # Weighted combination of current and historical
                        if assignment == most_common_team:
                            smoothed.append(assignment)
                        else:
                            # Stick with history unless very confident about change
                            smoothed.append(most_common_team)
                    continue

            smoothed.append(assignment)

        return smoothed

    def _update_team_profiles(self, features_list: List[Dict], assignments: List[int]):
        """Update team profiles with new observations."""
        team_features = defaultdict(list)

        for features, team in zip(features_list, assignments):
            feature_vector = []
            feature_vector.extend(features["jersey_mean"])
            feature_vector.extend(features["jersey_std"])
            feature_vector.extend(features["shorts_mean"])
            feature_vector.append(features["brightness"])
            feature_vector.append(features["contrast"])

            team_features[team].append(feature_vector)

        # Update profiles
        for team_id, features in team_features.items():
            if len(features) > 0:
                mean_features = np.mean(features, axis=0)

                if team_id in self.team_profiles:
                    # Exponential moving average for gradual updates
                    alpha = 0.1  # Learning rate
                    current_mean = np.array(
                        self.team_profiles[team_id]["mean_features"]
                    )
                    updated_mean = alpha * mean_features + (1 - alpha) * current_mean
                    self.team_profiles[team_id]["mean_features"] = updated_mean.tolist()
                else:
                    # Initialize new team profile
                    self.team_profiles[team_id] = {
                        "mean_features": mean_features.tolist(),
                        "sample_count": 1,
                    }


class TeamAssignmentProcessor(Processor):
    """Improved team assignment processor with temporal consistency."""

    def __init__(self, n_teams=2, consistency_weight=0.7):
        self.feature_extractor = TeamFeatureExtractor()
        self.assigner = TrackAwareTeamAssigner(
            n_teams=n_teams, consistency_weight=consistency_weight
        )

    def process(self, data: VideoData) -> VideoData:
        """Process video data with improved team assignment."""

        # Use progress bar for tracking
        progress_bar = tqdm(data.frames, desc="Improved team assignment", unit="frames")

        for frame_data in progress_bar:
            detections = frame_data.detections or []

            # Extract features and track IDs for players
            features_list = []
            track_ids = []
            player_indices = []

            for idx, detection in enumerate(detections):
                # Only process players and goalkeepers
                if getattr(detection, "object_type", None) in ("player", "goalkeeper"):
                    features = self.feature_extractor.extract(
                        frame_data.raw_frame, detection
                    )
                    track_id = (
                        detection.metadata.get("track_id", -1)
                        if detection.metadata
                        else -1
                    )

                    features_list.append(features)
                    track_ids.append(track_id)
                    player_indices.append(idx)

            # Assign teams
            if features_list:
                team_assignments = self.assigner.assign_teams(features_list, track_ids)

                # Apply assignments to detections
                for i, idx in enumerate(player_indices):
                    detection = detections[idx]
                    if detection.metadata is None:
                        detection.metadata = {}
                    detection.metadata["team"] = team_assignments[i]

        progress_bar.close()
        return data
