import numpy as np
import cv2
from typing import Dict, List, Any, Optional


class ViewTransformer:
    """
    A class for transforming coordinates from pixel space to real-world court coordinates.

    This class uses perspective transformation to map points from camera view to a top-down
    view of a football court with known dimensions.
    """

    def __init__(self) -> None:
        """
        Initialize the ViewTransformer with court dimensions and transformation matrix.

        Sets up the pixel vertices (camera view coordinates) and target vertices
        (real-world court coordinates) for perspective transformation.
        """
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array(
            [[110, 1035], [265, 275], [910, 260], [1640, 915]]
        )

        self.target_vertices = np.array(
            [[0, court_width], [0, 0], [court_length, 0], [court_length, court_width]]
        )

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform a point from pixel coordinates to real-world court coordinates.

        Args:
            point (np.ndarray): A 2D point in pixel coordinates [x, y]

        Returns:
            Optional[np.ndarray]: Transformed point in court coordinates, or None if point
                                is outside the transformation region
        """
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(
            reshaped_point, self.perspective_transformer
        )
        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(
        self, tracks: Dict[str, List[Dict[str, Dict[str, Any]]]]
    ) -> None:
        """
        Add transformed positions to tracking data for all objects across all frames.

        This method iterates through all tracked objects and frames, transforming their
        adjusted positions from pixel coordinates to court coordinates and adding the
        result as 'position_transformed' to each track.

        Args:
            tracks (Dict[str, List[Dict[str, Dict[str, Any]]]]): Nested dictionary structure containing:
                - object type (str) -> list of frames
                - each frame -> dict of track_id -> track_info
                - track_info contains 'position_adjusted' and will be updated with 'position_transformed'
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position_adjusted"]
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id][
                        "position_transformed"
                    ] = position_transformed
