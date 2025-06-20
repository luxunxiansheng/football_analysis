"""
Team Assignment Processor for robust player team classification.
"""

import numpy as np
from collections import defaultdict, Counter, deque
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

from ..domain.data_models import VideoData, FrameData
from ..domain.interfaces import Processor


class TeamFeatureExtractor:
    """Feature extractor focusing on proven techniques for team classification."""

    def extract(self, frame, detection) -> Dict[str, Any]:
        """Extract focused features for reliable team classification."""
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

        # Focus on jersey (upper body) and shorts (lower body)
        jersey_y_start = y1 + h // 6
        jersey_y_end = y1 + h // 2
        shorts_y_start = y1 + 2 * h // 3
        shorts_y_end = y2 - h // 10

        jersey_crop = frame[jersey_y_start:jersey_y_end, x1:x2]
        shorts_crop = frame[shorts_y_start:shorts_y_end, x1:x2]

        features = {}

        # 1. Jersey features (most important)
        if jersey_crop.size > 0:
            # Remove shadows/highlights for better color consistency
            jersey_flat = jersey_crop.reshape(-1, 3)
            brightness = np.mean(jersey_flat, axis=1)
            valid_mask = (brightness > 40) & (
                brightness < 200
            )  # Remove very dark/bright pixels

            if (
                np.sum(valid_mask) > jersey_flat.shape[0] * 0.1
            ):  # At least 10% valid pixels
                valid_jersey = jersey_flat[valid_mask]
                jersey_mean = np.mean(valid_jersey, axis=0)
                jersey_std = np.std(valid_jersey, axis=0)
            else:
                jersey_mean = np.mean(jersey_flat, axis=0)
                jersey_std = np.std(jersey_flat, axis=0)

            # Dominant color using median (more robust than mean)
            jersey_dominant = np.median(jersey_flat, axis=0)

            features.update(
                {
                    "jersey_mean": jersey_mean.tolist(),
                    "jersey_std": jersey_std.tolist(),
                    "jersey_dominant": jersey_dominant.tolist(),
                }
            )
        else:
            features.update(
                {
                    "jersey_mean": [0, 0, 0],
                    "jersey_std": [0, 0, 0],
                    "jersey_dominant": [0, 0, 0],
                }
            )

        # 2. Shorts features (secondary)
        if shorts_crop.size > 0:
            shorts_flat = shorts_crop.reshape(-1, 3)
            shorts_mean = np.mean(shorts_flat, axis=0)
            shorts_dominant = np.median(shorts_flat, axis=0)

            features.update(
                {
                    "shorts_mean": shorts_mean.tolist(),
                    "shorts_dominant": shorts_dominant.tolist(),
                }
            )
        else:
            features.update(
                {
                    "shorts_mean": [0, 0, 0],
                    "shorts_dominant": [0, 0, 0],
                }
            )

        # 3. Simple contrast measure
        if jersey_crop.size > 0:
            gray_jersey = np.mean(jersey_crop, axis=2)
            features["contrast"] = float(np.std(gray_jersey))
        else:
            features["contrast"] = 0.0

        return features


class TeamAssigner:
    """Optimized team assigner with focused consistency mechanisms."""

    def __init__(self, n_teams=2, stability_threshold=5, confidence_threshold=0.75):
        self.n_teams = n_teams
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold

        # Track assignment history with shorter memory for faster adaptation
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=20))

        # Team profiles with confidence tracking
        self.team_profiles: Dict[int, Dict] = {}
        self.profile_sample_counts: Dict[int, int] = defaultdict(int)

        # Stability tracking to prevent flickering
        self.track_stability: Dict[int, int] = defaultdict(int)
        self.last_assignments: Dict[int, int] = {}

        self.frame_count = 0

    def assign_teams(
        self, features_list: List[Dict], track_ids: List[int]
    ) -> List[int]:
        """Optimized team assignment with focused consistency."""

        if not features_list:
            return []

        self.frame_count += 1

        # Convert features to vectors
        feature_vectors = self._features_to_vectors(features_list)

        # Get base assignments
        if self.frame_count <= 10 or not self.team_profiles:
            # Bootstrap phase - use clustering
            base_assignments = self._bootstrap_clustering(feature_vectors)
        else:
            # Use profile-based assignment with track consistency
            base_assignments = self._profile_based_assignment(
                feature_vectors, track_ids
            )

        # Apply stability filter to prevent rapid switching
        final_assignments = self._apply_stability_filter(base_assignments, track_ids)

        # Update tracking structures
        self._update_structures(final_assignments, track_ids, features_list)

        return final_assignments

    def _features_to_vectors(self, features_list: List[Dict]) -> np.ndarray:
        """Convert features to numerical vectors with focused weighting."""
        vectors = []
        for features in features_list:
            vector = []

            # Primary features (jersey) - heavily weighted
            vector.extend(features["jersey_mean"])  # RGB values
            vector.extend(features["jersey_dominant"])  # Dominant color

            # Secondary features (shorts) - medium weight
            vector.extend([x * 0.7 for x in features["shorts_mean"]])

            # Tertiary features - low weight
            vector.append(features["contrast"] * 0.3)

            vectors.append(vector)

        return np.array(vectors)

    def _bootstrap_clustering(self, feature_vectors: np.ndarray) -> List[int]:
        """Bootstrap clustering with multiple attempts for stability."""
        if len(feature_vectors) < self.n_teams:
            return [1] * len(feature_vectors)

        # Try sklearn K-means first
        try:
            from sklearn.cluster import KMeans

            # Use multiple random states and pick the most stable result
            best_labels = None
            best_inertia = float("inf")

            for random_state in [42, 123, 456, 789]:
                try:
                    kmeans = KMeans(
                        n_clusters=self.n_teams, random_state=random_state, n_init=10
                    )
                    labels = kmeans.fit_predict(feature_vectors)

                    if kmeans.inertia_ < best_inertia:
                        best_inertia = kmeans.inertia_
                        best_labels = labels
                except Exception:
                    continue

            # If we got valid labels, return them
            if best_labels is not None:
                return [(label + 1) for label in best_labels]
            else:
                # Fallback to simple k-means if sklearn clustering failed
                return self._simple_kmeans(feature_vectors)

        except ImportError:
            # Fallback to simple k-means
            return self._simple_kmeans(feature_vectors)

    def _simple_kmeans(self, feature_vectors: np.ndarray) -> List[int]:
        """Simple k-means implementation."""
        n_samples = len(feature_vectors)
        if n_samples < self.n_teams:
            return [1] * n_samples

        # Multiple random initializations
        best_labels = None
        best_cost = float("inf")

        for seed in [42, 123, 456]:
            try:
                np.random.seed(seed)

                # Initialize centroids
                centroids = feature_vectors[
                    np.random.choice(n_samples, self.n_teams, replace=False)
                ]

                # K-means iterations
                for _ in range(15):
                    distances = np.linalg.norm(
                        feature_vectors[:, None] - centroids[None, :], axis=2
                    )
                    labels = distances.argmin(axis=1)

                    # Update centroids
                    for i in range(self.n_teams):
                        mask = labels == i
                        if np.any(mask):
                            centroids[i] = feature_vectors[mask].mean(axis=0)

                # Calculate total cost
                cost = sum(
                    np.min(np.linalg.norm(feature_vectors[i] - centroids, axis=1))
                    for i in range(n_samples)
                )

                if cost < best_cost:
                    best_cost = cost
                    best_labels = labels
            except Exception:
                continue

        # Return best labels if found, otherwise default assignment
        if best_labels is not None:
            return [(label + 1) for label in best_labels]
        else:
            # Ultimate fallback - assign alternating teams
            return [1 + (i % self.n_teams) for i in range(n_samples)]

    def _profile_based_assignment(
        self, feature_vectors: np.ndarray, track_ids: List[int]
    ) -> List[int]:
        """Assign teams based on learned profiles and track history."""
        assignments = []

        for i, (features, track_id) in enumerate(zip(feature_vectors, track_ids)):

            # First, check track history for strong patterns
            if track_id > 0 and track_id in self.track_history:
                history = list(self.track_history[track_id])
                if len(history) >= 3:
                    team_counts = Counter(history)
                    most_common_team, count = team_counts.most_common(1)[0]
                    confidence = count / len(history)

                    # If track has strong consistency, stick with it
                    if confidence >= self.confidence_threshold:
                        assignments.append(most_common_team)
                        continue

            # Use team profiles for assignment
            if self.team_profiles:
                similarities = []
                for team_id in range(1, self.n_teams + 1):
                    if team_id in self.team_profiles:
                        similarity = self._calculate_similarity(features, team_id)
                        similarities.append((team_id, similarity))

                if similarities:
                    # Weight similarity by profile confidence
                    weighted_similarities = []
                    for team_id, similarity in similarities:
                        profile_confidence = min(
                            1.0, self.profile_sample_counts[team_id] / 50.0
                        )
                        weighted_sim = similarity * (0.5 + 0.5 * profile_confidence)
                        weighted_similarities.append((team_id, weighted_sim))

                    best_team = max(weighted_similarities, key=lambda x: x[1])[0]
                    assignments.append(best_team)
                    continue

            # Fallback: assign to team 1
            assignments.append(1)

        return assignments

    def _calculate_similarity(self, features: np.ndarray, team_id: int) -> float:
        """Calculate similarity between features and team profile."""
        if team_id not in self.team_profiles:
            return 0.0

        profile_vector = np.array(self.team_profiles[team_id]["mean_features"])

        # Cosine similarity (more robust than Euclidean distance)
        dot_product = np.dot(features, profile_vector)
        norms = np.linalg.norm(features) * np.linalg.norm(profile_vector)

        if norms > 0:
            similarity = dot_product / norms
            return max(0.0, float(similarity))  # Clamp to positive
        else:
            return 0.0

    def _apply_stability_filter(
        self, assignments: List[int], track_ids: List[int]
    ) -> List[int]:
        """Apply stability filter to prevent rapid assignment changes."""
        filtered_assignments = []

        for assignment, track_id in zip(assignments, track_ids):
            if track_id > 0:
                last_assignment = self.last_assignments.get(track_id)

                if last_assignment is not None:
                    if assignment == last_assignment:
                        # Same assignment - increase stability
                        self.track_stability[track_id] += 1
                        filtered_assignments.append(assignment)
                    else:
                        # Different assignment - check if we should change
                        current_stability = self.track_stability.get(track_id, 0)

                        if current_stability < self.stability_threshold:
                            # Not stable enough - keep previous assignment
                            filtered_assignments.append(last_assignment)
                        else:
                            # Stable enough - allow change but reset stability
                            filtered_assignments.append(assignment)
                            self.track_stability[track_id] = 1
                else:
                    # First assignment for this track
                    filtered_assignments.append(assignment)
                    self.track_stability[track_id] = 1
            else:
                # Untracked detection - use current assignment
                filtered_assignments.append(assignment)

        return filtered_assignments

    def _update_structures(
        self, assignments: List[int], track_ids: List[int], features_list: List[Dict]
    ):
        """Update tracking structures and team profiles."""

        # Update track history and last assignments
        for assignment, track_id, features in zip(
            assignments, track_ids, features_list
        ):
            if track_id > 0:
                self.track_history[track_id].append(assignment)
                self.last_assignments[track_id] = assignment

        # Update team profiles with stable tracks only
        team_features = defaultdict(list)
        for assignment, track_id, features in zip(
            assignments, track_ids, features_list
        ):
            # Only use features from stable tracks for profile updates
            if track_id > 0 and self.track_stability.get(track_id, 0) >= 3:
                feature_vector = self._features_to_vectors([features])[0]
                team_features[assignment].append(feature_vector)

        # Update profiles with exponential moving average
        for team_id, features in team_features.items():
            if len(features) > 0:
                new_mean = np.mean(features, axis=0)

                if team_id in self.team_profiles:
                    # Exponential moving average
                    alpha = 0.1  # Conservative learning rate
                    current_mean = np.array(
                        self.team_profiles[team_id]["mean_features"]
                    )
                    updated_mean = alpha * new_mean + (1 - alpha) * current_mean
                    self.team_profiles[team_id]["mean_features"] = updated_mean.tolist()
                else:
                    # Initialize new profile
                    self.team_profiles[team_id] = {"mean_features": new_mean.tolist()}

                self.profile_sample_counts[team_id] += len(features)


class TeamAssignmentProcessor(Processor):
    """Optimized team assignment processor with focused improvements."""

    def __init__(self, n_teams=2, stability_threshold=5, confidence_threshold=0.75):
        self.feature_extractor = TeamFeatureExtractor()
        self.assigner = TeamAssigner(
            n_teams=n_teams,
            stability_threshold=stability_threshold,
            confidence_threshold=confidence_threshold,
        )

    def process(self, data: VideoData) -> VideoData:
        """Process video data with optimized team assignment."""

        progress_bar = tqdm(
            data.frames, desc="Optimized team assignment", unit="frames"
        )

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
