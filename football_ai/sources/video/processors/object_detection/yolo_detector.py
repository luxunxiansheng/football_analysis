from football_ai.utilities import np, create_progress_bar
from football_ai.core_models.video import Video
from football_ai.core_models.constants import ObjectType
from football_ai.core_models.interfaces import Processor
from football_ai.core_models import Player, Goalkeeper, Referee, Ball

from ultralytics import YOLO


class ObjectDetectionProcessor(Processor):
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.1,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        max_detections: int = 1000,
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.max_detections = max_detections

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
        """Parse YOLO results into domain objects."""
        objects = []
        if yolo_result.boxes is None:
            return objects
        
        class_names = yolo_result.names
        boxes = yolo_result.boxes.xyxy.cpu().numpy()
        confidences = yolo_result.boxes.conf.cpu().numpy()
        class_ids = yolo_result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            class_name = class_names.get(class_id, "unknown")
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            confidence = float(conf)
            
            # Create appropriate domain object based on detection type
            if class_name == "player":
                obj = Player(
                    object_id=None,  # Will be assigned during tracking
                    bounding_box=(x1, y1, x2, y2),
                    confidence=confidence,
                    pixel_position=((x1 + x2) / 2, (y1 + y2) / 2),
                )
            elif class_name == "goalkeeper":
                obj = Goalkeeper(
                    object_id=None,
                    bounding_box=(x1, y1, x2, y2),
                    confidence=confidence,
                    pixel_position=((x1 + x2) / 2, (y1 + y2) / 2),
                )
            elif class_name == "referee":
                obj = Referee(
                    object_id=None,
                    bounding_box=(x1, y1, x2, y2),
                    confidence=confidence,
                    pixel_position=((x1 + x2) / 2, (y1 + y2) / 2),
                )
            elif class_name == "ball":
                obj = Ball(
                    object_id=None,
                    bounding_box=(x1, y1, x2, y2),
                    confidence=confidence,
                    pixel_position=((x1 + x2) / 2, (y1 + y2) / 2),
                )
            else:
                continue
                
            objects.append(obj)
        return objects
