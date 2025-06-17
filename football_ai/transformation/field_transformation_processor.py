from ..domain.data_models import VideoData, FrameData
from ..domain.interfaces import Processor
from typing import Tuple
import numpy as np
import cv2


class FieldTransformationProcessor(Processor):
    def __init__(
        self,
        field_width: float = 105.0,
        field_height: float = 68.0,
        pixel_corners: list = None,
    ):
        if pixel_corners is None:
            # Default: user-specified real field corners
            pixel_corners = [
                [110, 1035],
                [265, 275],
                [910, 260],
                [1640, 915],
            ]
        self.field_width = field_width
        self.field_height = field_height
        self.pixel_corners = np.array(pixel_corners, dtype=np.float32)
        self.field_corners = np.array(
            [
                [0, 0],
                [field_width, 0],
                [field_width, field_height],
                [0, field_height],
            ],
            dtype=np.float32,
        )
        self.perspective_matrix = cv2.getPerspectiveTransform(
            self.pixel_corners, self.field_corners
        )

    def process(self, data: VideoData) -> VideoData:
        for frame_data in data.frames:
            detections = frame_data.detections or []
            for detection in detections:
                if detection.metadata is None:
                    detection.metadata = {}
                pixel_pos = detection.metadata.get("object_position")
                if pixel_pos is not None:
                    field_pos = self.transform_point(tuple(pixel_pos))
                    detection.metadata["field_position"] = field_pos
        return data

    def transform_point(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        point = np.array([[pixel_point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.perspective_matrix)
        x, y = transformed[0, 0]
        x = max(0, min(x, self.field_width))
        y = max(0, min(y, self.field_height))
        return (float(x), float(y))
