"""
YOLO-based keypoint detector for player pose estimation in football analysis.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

from ..domain.interfaces import KeypointDetector
from ..domain.models import Detection, BoundingBox, ObjectType, PlayerKeypoints


class YOLOKeypointDetector(KeypointDetector):
    """YOLO pose estimation detector implementation."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """Initialize the YOLO keypoint detector."""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_keypoints(self, frame: np.ndarray) -> List[PlayerKeypoints]:
        """Detect pose keypoints for all players in a single frame."""
        results = self.model(frame, verbose=False)
        keypoints_list = []

        for result in results:
            if hasattr(result, "keypoints") and result.keypoints is not None:
                # Extract keypoints data
                keypoints_data = result.keypoints.data  # Shape: [N, num_keypoints, 3]
                boxes_data = result.boxes.data if result.boxes is not None else None

                for i, kpts in enumerate(keypoints_data):
                    # Get corresponding bounding box if available
                    bbox = None
                    if boxes_data is not None and i < len(boxes_data):
                        box = boxes_data[i]
                        bbox = BoundingBox(
                            x1=float(box[0]),
                            y1=float(box[1]),
                            x2=float(box[2]),
                            y2=float(box[3]),
                            confidence=float(box[4]),
                        )
                    else:
                        # Create bbox from keypoints if no detection box available
                        visible_kpts = [
                            (x, y) for x, y, c in kpts if c > self.confidence_threshold
                        ]
                        if visible_kpts:
                            x_coords = [x for x, y in visible_kpts]
                            y_coords = [y for x, y in visible_kpts]
                            bbox = BoundingBox(
                                x1=min(x_coords),
                                y1=min(y_coords),
                                x2=max(x_coords),
                                y2=max(y_coords),
                                confidence=0.8,
                            )

                    if bbox is not None:
                        # Convert keypoints to list of tuples
                        keypoint_tuples = [
                            (float(x), float(y), float(c)) for x, y, c in kpts
                        ]

                        player_keypoints = PlayerKeypoints(
                            keypoints=keypoint_tuples,
                            bbox=bbox,
                            confidence=bbox.confidence,
                        )
                        keypoints_list.append(player_keypoints)

        return keypoints_list

    def detect_keypoints_from_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> List[PlayerKeypoints]:
        """Detect pose keypoints for specific player detections."""
        keypoints_list = []

        # Run keypoint detection on the full frame
        all_keypoints = self.detect_keypoints(frame)

        # Match keypoints to detections based on bounding box overlap
        for detection in detections:
            if detection.object_type in [ObjectType.PLAYER, ObjectType.GOALKEEPER]:
                best_match = None
                best_iou = 0.0

                for keypoints in all_keypoints:
                    iou = self._calculate_iou(detection.bbox, keypoints.bbox)
                    if iou > best_iou and iou > 0.3:  # Minimum overlap threshold
                        best_iou = iou
                        best_match = keypoints

                if best_match:
                    # Update the keypoints with detection information
                    best_match.track_id = detection.track_id
                    keypoints_list.append(best_match)

        return keypoints_list

    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        # Calculate intersection area
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)

        if x1 >= x2 or y1 >= y2:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union area
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def detect_keypoints_batch(
        self, frames: List[np.ndarray], batch_size: int = 8
    ) -> List[List[PlayerKeypoints]]:
        """Detect keypoints in multiple frames efficiently."""
        all_keypoints = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch_results = self.model(batch_frames, verbose=False)

            for result in batch_results:
                frame_keypoints = []

                if hasattr(result, "keypoints") and result.keypoints is not None:
                    keypoints_data = result.keypoints.data
                    boxes_data = result.boxes.data if result.boxes is not None else None

                    for j, kpts in enumerate(keypoints_data):
                        bbox = None
                        if boxes_data is not None and j < len(boxes_data):
                            box = boxes_data[j]
                            bbox = BoundingBox(
                                x1=float(box[0]),
                                y1=float(box[1]),
                                x2=float(box[2]),
                                y2=float(box[3]),
                                confidence=float(box[4]),
                            )

                        if bbox is not None:
                            keypoint_tuples = [
                                (float(x), float(y), float(c)) for x, y, c in kpts
                            ]

                            player_keypoints = PlayerKeypoints(
                                keypoints=keypoint_tuples,
                                bbox=bbox,
                                confidence=bbox.confidence,
                            )
                            frame_keypoints.append(player_keypoints)

                all_keypoints.append(frame_keypoints)

        return all_keypoints
