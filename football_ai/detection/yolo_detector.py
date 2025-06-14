"""
Modern YOLO-based object detector for football analysis.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

try:
    import supervision as sv
except ImportError:
    sv = None

from ..domain.interfaces import ObjectDetector
from ..domain.models import Detection, BoundingBox, ObjectType


class YOLODetector(ObjectDetector):
    """YOLO detector implementation with clean architecture."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.1):
        """Initialize the YOLO detector."""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_objects_in_frames(
        self, frames: List[np.ndarray], batch_size: int = 20
    ) -> List[List[Detection]]:
        """Detect objects in video frames."""
        all_detections = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch_results = self.model.predict(
                batch_frames, conf=self.confidence_threshold
            )

            for result in batch_results:
                frame_detections = self._parse_yolo_result(result)
                all_detections.append(frame_detections)

        return all_detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame."""
        result = self.model.predict(
            frame, conf=self.confidence_threshold, verbose=False
        )
        return self._parse_yolo_result(result[0])

    def _parse_yolo_result(self, yolo_result) -> List[Detection]:
        """Parse YOLO result into Detection objects."""
        detections = []

        if yolo_result.boxes is None:
            return detections

        class_names = yolo_result.names
        boxes = yolo_result.boxes.xyxy.cpu().numpy()
        confidences = yolo_result.boxes.conf.cpu().numpy()
        class_ids = yolo_result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            class_name = class_names.get(class_id, "unknown")

            # Map class names to our ObjectType enum
            if class_name == "player":
                object_type = ObjectType.PLAYER
            elif class_name == "goalkeeper":
                object_type = ObjectType.GOALKEEPER
            elif class_name == "referee":
                object_type = ObjectType.REFEREE
            elif class_name == "ball":
                object_type = ObjectType.BALL
            else:
                continue  # Skip unknown classes

            bbox = BoundingBox(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                confidence=float(conf),
            )

            detection = Detection(
                bbox=bbox, object_type=object_type, confidence=float(conf)
            )

            detections.append(detection)

        return detections
