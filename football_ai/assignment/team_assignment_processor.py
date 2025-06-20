"""
Team Assignment Processor for robust player team classification.
"""

import numpy as np
from collections import defaultdict, Counter, deque
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from more_itertools import chunked

import torch
from transformers import AutoProcessor, SiglipVisionModel
import umap
from sklearn.cluster import KMeans

from football_ai import detection

from ..domain.data_models import ObjectType, VideoData, FrameData
from ..domain.interfaces import Processor


class SigLIPTeamAssignmentProcessor(Processor):
    def __init__(self, model_path: str, data: VideoData):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.embbedding_model = SiglipVisionModel.from_pretrained(model_path).to(DEVICE)
        self.embedding_processor = AutoProcessor.from_pretrained(model_path)

        cropped_players = self._extract_players(data)
        BATCH_SIZE = 32

        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = chunked(crops, BATCH_SIZE)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='embedding extraction'):
                inputs = self.embedding_processor(images=batch, return_tensors="pt").to(DEVICE)
                outputs = self.embbedding_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        data = np.concatenate(data)

        REDUCER = umap.UMAP(n_components=3)
        CLUSTERING_MODEL = KMeans(n_clusters=2)

        projections = REDUCER.fit_transform(data)
        clusters = CLUSTERING_MODEL.fit_predict(projections)


    def _extract_players(self, data):
        cropped_players = []
        for frame_data in data.frames:
            frame = frame_data.raw_frame
            detections = frame_data.detections
            if detections is not None:
                for detection in detections:
                    if detection.object_type == ObjectType.PLAYER:
                        x1, y1, x2, y2 = map(
                            int,
                            [
                                detection.bbox.x1,
                                detection.bbox.y1,
                                detection.bbox.x2,
                                detection.bbox.y2,
                            ],
                        )

                        cropped_player = frame[y1:y2, x1:x2]
                        cropped_players.append(cropped_player)

        return cropped_players

    def process(self, data: VideoData) -> VideoData:

        return data
