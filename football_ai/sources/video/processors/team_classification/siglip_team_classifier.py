from football_ai.utilities import List, Optional, os, np, torch, create_progress_bar
import umap
from PIL import Image
from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel
from tqdm import tqdm

from football_ai.core_models.interfaces import Processor
from football_ai.core_models.video import Video
from football_ai.core_models.constants import ObjectType


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
                # Move model to device (ignore type checker warning)
                self.model = self.model.to(self.device)  # type: ignore
                self.processor = AutoProcessor.from_pretrained(self.model_path)
            else:
                self.model = SiglipVisionModel.from_pretrained(model_name)
                # Move model to device (ignore type checker warning)
                self.model = self.model.to(self.device)  # type: ignore
                self.processor = AutoProcessor.from_pretrained(model_name)

                if self.model_path:
                    os.makedirs(self.model_path, exist_ok=True)
                    self.model.save_pretrained(self.model_path)
                    self.processor.save_pretrained(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load SigLIP model: {e}")

    def _extract_crops(self, video_data: Video) -> List[np.ndarray]:
        """Extract player and goalkeeper crops from video frames using new Frame model."""
        crops = []
        frame_progress = tqdm(
            video_data.frames, desc="Extracting player crops", unit="frames"
        )
        for frame in frame_progress:
            if frame.raw_frame is None:
                continue
            frame_crops = 0
            for player in list(frame.players.values()) + list(
                frame.goalkeepers.values()
            ):
                if (
                    not hasattr(player, "pixel_position")
                    or player.pixel_position is None
                ):
                    continue
                cx, cy = map(int, player.pixel_position)
                crop_size = 32  # Example size, adjust as needed
                x1, y1 = max(0, cx - crop_size), max(0, cy - crop_size)
                x2, y2 = cx + crop_size, cy + crop_size
                h, w = frame.raw_frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame.raw_frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops.append(crop)
                        frame_crops += 1
            frame_progress.set_postfix(
                {"Total crops": len(crops), "Frame crops": frame_crops}
            )
        frame_progress.close()
        print(
            f"✅ Extracted {len(crops)} player crops from {len(video_data.frames)} frames"
        )
        return crops

    def _extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract features using SigLIP model."""
        if not crops:
            return np.array([])

        # Convert to PIL and process in batches with progress bar
        features = []
        num_batches = (len(crops) + self.batch_size - 1) // self.batch_size

        print(f"🧠 Extracting features from {len(crops)} crops using SigLIP model...")
        batch_progress = tqdm(
            range(0, len(crops), self.batch_size),
            desc="Feature extraction",
            unit="batch",
            total=num_batches,
        )

        with torch.no_grad():
            for i in batch_progress:
                batch_crops = crops[i : i + self.batch_size]
                batch_pil = [
                    Image.fromarray(crop[..., ::-1]) for crop in batch_crops
                ]  # BGR to RGB

                inputs = self.processor(images=batch_pil, return_tensors="pt").to(
                    self.device
                )
                outputs = self.model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                features.append(embeddings)

                # Update progress with current stats
                batch_progress.set_postfix(
                    {
                        "Batch size": len(batch_crops),
                        "Features extracted": len(features) * self.batch_size,
                    }
                )

        batch_progress.close()

        result = np.concatenate(features) if features else np.array([])
        print(
            f"✅ Extracted {result.shape[0]} feature vectors of dimension {result.shape[1] if len(result.shape) > 1 else 0}"
        )
        return result

    def train(self, video_data: Video, batch_size: Optional[int] = None) -> None:
        """Train the team assignment model."""
        print("🏋️ Training SigLIP Team Assignment Model...")

        if batch_size:
            self.batch_size = batch_size

        # Step 1: Extract crops with progress
        crops = self._extract_crops(video_data)
        if not crops:
            raise ValueError("No player crops found")

        # Step 2: Extract features with progress
        features = self._extract_features(crops)

        # Step 3: Dimensionality reduction with UMAP
        print("🔄 Reducing dimensions with UMAP...")
        with tqdm(total=1, desc="UMAP reduction") as pbar:
            projections = self.reducer.fit_transform(features)
            pbar.update(1)
        print("✅ UMAP dimensionality reduction completed")

        # Step 4: Clustering with KMeans
        print("🎯 Training KMeans clustering...")
        with tqdm(total=1, desc="KMeans training") as pbar:
            self.cluster_model.fit(projections)
            pbar.update(1)

        self._is_trained = True
        print("✅ SigLIP Team Assignment Model training completed!")
        print(f"📊 Model trained on {len(crops)} player crops")
        print(f"🏆 Ready to assign players to 2 teams")

    def process(self, video_data: Video) -> Video:
        """Assign team labels to players and goalkeepers using SigLIP model."""
        if not self._is_trained:
            print("⚠️ Model not trained yet. Training on provided data...")
            self.train(video_data)
        print("🏃‍♂️ Assigning team labels to players...")
        frame_progress = tqdm(
            video_data.frames, desc="Processing frames", unit="frames"
        )
        total_assignments = 0
        for frame in frame_progress:
            if frame.raw_frame is None:
                continue
            player_crops, player_objs = [], []
            for player in list(frame.players.values()) + list(
                frame.goalkeepers.values()
            ):
                if (
                    not hasattr(player, "pixel_position")
                    or player.pixel_position is None
                ):
                    continue
                cx, cy = map(int, player.pixel_position)
                crop_size = 32
                x1, y1 = max(0, cx - crop_size), max(0, cy - crop_size)
                x2, y2 = cx + crop_size, cy + crop_size
                h, w = frame.raw_frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame.raw_frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        player_crops.append(crop)
                        player_objs.append(player)
            if player_crops:
                player_features = self._extract_features_silent(player_crops)
                player_projections = self.reducer.transform(player_features)
                team_labels = self.cluster_model.predict(player_projections)
                frame_assignments = 0
                for player_obj, team_id in zip(player_objs, team_labels):
                    player_obj.team_id = int(team_id)
                    frame_assignments += 1
                    total_assignments += 1
                frame_progress.set_postfix(
                    {
                        "Frame players": frame_assignments,
                        "Total assigned": total_assignments,
                    }
                )
        frame_progress.close()
        print(f"✅ Team assignment completed!")
        print(
            f"👥 Assigned {total_assignments} players to teams across {len(video_data.frames)} frames"
        )
        return video_data

    def _extract_features_silent(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract features without progress bars (for single frame processing)."""
        if not crops:
            return np.array([])

        features = []
        with torch.no_grad():
            for i in range(0, len(crops), self.batch_size):
                batch_crops = crops[i : i + self.batch_size]
                batch_pil = [
                    Image.fromarray(crop[..., ::-1]) for crop in batch_crops
                ]  # BGR to RGB

                inputs = self.processor(images=batch_pil, return_tensors="pt").to(
                    self.device
                )
                outputs = self.model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                features.append(embeddings)

        return np.concatenate(features) if features else np.array([])
