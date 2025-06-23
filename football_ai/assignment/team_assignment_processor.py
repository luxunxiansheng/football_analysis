from typing import List, Optional
import os

import numpy as np
import torch
import umap
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

from ..domain.interfaces import Processor
from ..domain.data_models import VideoData, ObjectType


class SigLIPTeamAssignmentProcessor(Processor):
    """Team assignment using SigLIP vision model + UMAP + KMeans clustering."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 16,
        n_clusters: int = 2,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model_path = model_path
        self._is_trained = False

        self._load_model()
        self.reducer = umap.UMAP(n_components=3, random_state=42)
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)

    def _load_model(self) -> None:
        """Load SigLIP model and processor."""
        model_name = "google/siglip-base-patch16-224"

        try:
            if self.model_path and os.path.exists(self.model_path):
                self.model = SiglipVisionModel.from_pretrained(self.model_path)
                self.model = self.model.to(self.device)
                self.processor = AutoProcessor.from_pretrained(self.model_path)
            else:
                self.model = SiglipVisionModel.from_pretrained(model_name)
                self.model = self.model.to(self.device)
                self.processor = AutoProcessor.from_pretrained(model_name)

                if self.model_path:
                    os.makedirs(self.model_path, exist_ok=True)
                    self.model.save_pretrained(self.model_path)
                    self.processor.save_pretrained(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load SigLIP model: {e}")

    def _extract_crops(self, video_data: VideoData) -> List[np.ndarray]:
        """Extract player crops from video frames."""
        crops = []
        for frame in video_data.frames:
            if not frame.detections:
                continue

            for detection in frame.detections:
                if detection.object_type != ObjectType.PLAYER:
                    continue

                bbox = detection.bbox
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                h, w = frame.raw_frame.shape[:2]

                # Validate and clip bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    crop = frame.raw_frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops.append(crop)
        return crops

    def _extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract features using SigLIP model."""
        if not crops:
            return np.array([])

        # Convert to PIL and process in batches
        features = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(crops), self.batch_size), desc="Extracting features"
            ):
                batch_crops = crops[i : i + self.batch_size]
                batch_pil = [
                    Image.fromarray(crop[..., ::-1]) for crop in batch_crops
                ]  # BGR to RGB

                inputs = self.processor(images=batch_pil, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                features.append(embeddings)

        return np.concatenate(features) if features else np.array([])

    def train(self, video_data: VideoData, batch_size: Optional[int] = None) -> None:
        """Train the team assignment model."""
        if batch_size:
            self.batch_size = batch_size

        crops = self._extract_crops(video_data)
        if not crops:
            raise ValueError("No player crops found")

        features = self._extract_features(crops)
        projections = self.reducer.fit_transform(features)
        self.cluster_model.fit(projections)
        self._is_trained = True

    def process(self, video_data: VideoData) -> VideoData:
        """Assign team labels to players."""
        if not self._is_trained:
            self.train(video_data)
              

        for frame in video_data.frames:
            if not frame.detections:
                continue

            player_crops, player_detections = [], []

            for detection in frame.detections:
                if detection.object_type != ObjectType.PLAYER:
                    continue

                bbox = detection.bbox
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                h, w = frame.raw_frame.shape[:2]

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    crop = frame.raw_frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        player_crops.append(crop)
                        player_detections.append(detection)

            if player_crops:
                player_features = self._extract_features(player_crops)
                player_projections = self.reducer.transform(player_features)
                team_labels = self.cluster_model.predict(player_projections)

                for detection, team_id in zip(player_detections, team_labels):
                    if detection.metadata is None:
                        detection.metadata = {}
                    detection.metadata["team"] = int(team_id)

        return video_data
