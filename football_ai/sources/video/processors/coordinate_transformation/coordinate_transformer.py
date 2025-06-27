from football_ai.core_models import Video, Frame
from football_ai.core_models.interfaces import Processor
from typing import Tuple, Optional
import numpy as np
import cv2


class FieldTransformationProcessor(Processor):
    def __init__(
        self,
        field_width: float = 105.0,
        field_height: float = 68.0,
        pixel_corners: Optional[list] = None,
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

    def process(self, data: Video) -> Video:
        for frame in data.frames:
            # Transform all tracked objects
            for obj in (
                list(frame.players.values())
                + list(frame.goalkeepers.values())
                + list(frame.referees.values())
            ):
                if obj.pixel_position is not None:
                    field_pos = self.transform_point(tuple(obj.pixel_position))
                    obj.field_position = field_pos
            # Optionally, transform ball position
            if frame.ball and frame.ball.pixel_position is not None:
                field_pos = self.transform_point(tuple(frame.ball.pixel_position))
                frame.ball.field_position = field_pos
        return data

    def transform_point(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        point = np.array([[pixel_point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.perspective_matrix)
        x, y = transformed[0, 0]
        x = max(0, min(x, self.field_width))
        y = max(0, min(y, self.field_height))
        return (float(x), float(y))
