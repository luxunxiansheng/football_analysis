from football_ai.utils import np, create_progress_bar
from football_ai.domain.video import Video
from football_ai.domain.constants import ObjectType
from football_ai.domain.interfaces import Processor

from ultralytics import YOLO

# Temporary classes for backward compatibility
from dataclasses import dataclass
from typing import Optional


@dataclass
class BoundingBox:
    """Temporary BoundingBox for backward compatibility."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: Optional[float] = None


@dataclass
class Detection:
    """Temporary Detection for backward compatibility."""

    bbox: BoundingBox
    object_type: str
    confidence: float


class ObjectDetectionProcessor(Processor):
    def __init__(self, model_path: str, confidence_threshold: float = 0.1):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def process(self, data: Video) -> Video:
        # Use progress bar for object detection
        frames_progress_bar = create_progress_bar(
            iterable=data.frames, desc="Object detection", unit="frames"
        )

        for frame_data in frames_progress_bar:
            frame = frame_data.raw_frame
            if frame is not None:
                detections = self._detect_objects(frame)
                frame_data.detections = detections

        frames_progress_bar.close()
        return data

    def _detect_objects(self, frame: np.ndarray) -> list:
        result = self.model.predict(
            frame, conf=self.confidence_threshold, verbose=False
        )
        return self._parse_yolo_result(result[0])

    def _parse_yolo_result(self, yolo_result) -> list:
        detections = []
        if yolo_result.boxes is None:
            return detections
        class_names = yolo_result.names
        boxes = yolo_result.boxes.xyxy.cpu().numpy()
        confidences = yolo_result.boxes.conf.cpu().numpy()
        class_ids = yolo_result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            class_name = class_names.get(class_id, "unknown")
            if class_name == "player":
                object_type = ObjectType.PLAYER
            elif class_name == "goalkeeper":
                object_type = ObjectType.GOALKEEPER
            elif class_name == "referee":
                object_type = ObjectType.REFEREE
            elif class_name == "ball":
                object_type = ObjectType.BALL
            else:
                continue
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
