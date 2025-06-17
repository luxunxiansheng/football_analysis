import numpy as np
from tqdm import tqdm
from ..domain.data_models import VideoData, FrameData
from ..domain.interfaces import Processor
from typing import Any


class TeamFeatureExtractor:
    def extract(self, frame: Any, detection) -> Any:
        # Extract mean color from the top half of the detection's bounding box
        x1, y1, x2, y2 = map(
            int,
            [
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2,
            ],
        )
        h = y2 - y1
        y2_top = y1 + h // 2
        crop = frame[y1:y2_top, x1:x2]
        if crop.size == 0:
            return [0, 0, 0]
        mean_color = crop.mean(axis=(0, 1))
        return mean_color.tolist()


class KMeansTeamAssigner:
    def __init__(self, n_teams=2):
        self.n_teams = n_teams

    def assign(self, features: np.ndarray) -> np.ndarray:
        np.random.seed(0)
        centroids = features[
            np.random.choice(len(features), self.n_teams, replace=False)
        ]
        for _ in range(10):
            dists = np.linalg.norm(features[:, None] - centroids[None, :], axis=2)
            labels = dists.argmin(axis=1)
            for i in range(self.n_teams):
                if np.any(labels == i):
                    centroids[i] = features[labels == i].mean(axis=0)
        return labels


class TeamAssignmentProcessor(Processor):
    def __init__(self, feature_extractor=None, assigner=None):
        self.feature_extractor = feature_extractor or TeamFeatureExtractor()
        self.assigner = assigner or KMeansTeamAssigner()

    def process(self, data: VideoData) -> VideoData:
        # Use progress bar only if processing many frames (>50)
        if len(data.frames) > 50:
            progress_bar = tqdm(data.frames, desc="Team assignment", unit="frames")
            frame_iterator = progress_bar
        else:
            frame_iterator = data.frames
            progress_bar = None

        for frame_data in frame_iterator:
            detections = frame_data.detections or []
            features = []
            player_indices = []
            for idx, detection in enumerate(detections):
                # Assign to team if detection is a player or goalkeeper by class name
                if getattr(detection, "object_type", None) in (
                    "player",
                    "goalkeeper",
                ) or getattr(detection, "class_name", "").lower() in (
                    "player",
                    "goalkeeper",
                ):
                    feat = self.feature_extractor.extract(
                        frame_data.raw_frame, detection
                    )
                    features.append(feat)
                    player_indices.append(idx)
            if features:
                features_np = np.array(features)
                labels = self.assigner.assign(features_np)
                for i, idx in enumerate(player_indices):
                    detection = detections[idx]
                    if detection.metadata is None:
                        detection.metadata = {}
                    detection.metadata["team"] = (
                        int(labels[i]) + 1
                    )  # Use integer team id (1, 2, ...)

        if progress_bar:
            progress_bar.close()
        return data
