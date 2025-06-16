from ..domain.data_models import VideoData, FrameData
from ..domain.interfaces import Processor
from typing import Tuple
import numpy as np
import cv2


class SimpleFieldTransformationProcessor(Processor):
    def __init__(self, field_width: float, field_height: float, pixel_corners: list):
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
            positions = (
                frame_data.metadata.get("object_positions", {})
                if frame_data.metadata
                else {}
            )
            field_positions = {}
            for obj_id, pixel_pos in positions.items():
                field_pos = self.transform_point(tuple(pixel_pos))
                field_positions[obj_id] = field_pos
            if frame_data.metadata is None:
                frame_data.metadata = {}
            frame_data.metadata["field_positions"] = field_positions
        return data

    def transform_point(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        point = np.array([[pixel_point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.perspective_matrix)
        x, y = transformed[0, 0]
        x = max(0, min(x, self.field_width))
        y = max(0, min(y, self.field_height))
        return (float(x), float(y))
