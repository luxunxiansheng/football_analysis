"""
Coordinate Transformation Module

This module provides functionality to transform pixel coordinates to real-world
field coordinates using perspective transformation and homography.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from ..domain.interfaces import CoordinateTransformer


class PerspectiveCoordinateTransformer(CoordinateTransformer):
    """
    Implementation of coordinate transformation using perspective
    transformation to map pixel coordinates to real football field coordinates.
    """

    def __init__(
        self, field_width: float = 68.0, field_height: float = 105.0  # meters
    ):  # meters
        """
        Initialize the coordinate transformer.

        Args:
            field_width: Width of football field in meters
            field_height: Height of football field in meters
        """
        self.field_width = field_width
        self.field_height = field_height

        # Transformation matrices
        self.perspective_matrix: Optional[np.ndarray] = None
        self.inverse_perspective_matrix: Optional[np.ndarray] = None

        # Field corners in pixel coordinates (will be set during calibration)
        self.pixel_vertices: Optional[List[List[float]]] = None

        # Default field vertices in real-world coordinates (meters)
        self.field_vertices = [
            [0, 0],  # Top-left corner
            [self.field_width, 0],  # Top-right corner
            [self.field_width, self.field_height],  # Bottom-right corner
            [0, self.field_height],  # Bottom-left corner
        ]

    def calibrate(self, pixel_vertices: List[List[float]]) -> bool:
        """
        Calibrate the transformer using field corner coordinates in pixels.

        Args:
            pixel_vertices: List of [x, y] pixel coordinates for field corners
                           in order: [top-left, top-right, bottom-right, bottom-left]

        Returns:
            True if calibration successful, False otherwise
        """
        try:
            if len(pixel_vertices) != 4:
                print("Error: Need exactly 4 corner points for calibration")
                return False

            self.pixel_vertices = pixel_vertices

            # Convert to numpy arrays
            src_points = np.array(pixel_vertices, dtype=np.float32)
            dst_points = np.array(self.field_vertices, dtype=np.float32)

            # Calculate perspective transformation matrix
            self.perspective_matrix = cv2.getPerspectiveTransform(
                src_points, dst_points
            )
            self.inverse_perspective_matrix = cv2.getPerspectiveTransform(
                dst_points, src_points
            )

            print("Coordinate transformer calibrated successfully")
            return True

        except Exception as e:
            print(f"Error calibrating coordinate transformer: {e}")
            return False

    def transform_point(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a pixel coordinate to real-world field coordinate.

        Args:
            pixel_point: (x, y) coordinates in pixels

        Returns:
            (x, y) coordinates in meters on the field
        """
        if self.perspective_matrix is None:
            # If not calibrated, return scaled coordinates as fallback
            return self._fallback_transform(pixel_point)

        try:
            # Convert point to homogeneous coordinates
            point = np.array([[pixel_point[0], pixel_point[1]]], dtype=np.float32)

            # Apply perspective transformation
            transformed = cv2.perspectiveTransform(
                point.reshape(1, 1, 2), self.perspective_matrix
            )

            # Extract coordinates
            x, y = transformed[0, 0]

            # Clamp to field boundaries
            x = max(0, min(x, self.field_width))
            y = max(0, min(y, self.field_height))

            return (float(x), float(y))

        except Exception as e:
            print(f"Error transforming point: {e}")
            return self._fallback_transform(pixel_point)

    def transform_points(
        self, pixel_points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Transform multiple pixel coordinates to real-world field coordinates.

        Args:
            pixel_points: List of (x, y) coordinates in pixels

        Returns:
            List of (x, y) coordinates in meters on the field
        """
        if not pixel_points:
            return []

        if self.perspective_matrix is None:
            return [self._fallback_transform(point) for point in pixel_points]

        try:
            # Convert points to numpy array
            points_array = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)

            # Apply perspective transformation
            transformed = cv2.perspectiveTransform(
                points_array, self.perspective_matrix
            )

            # Extract and clamp coordinates
            result = []
            for point in transformed.reshape(-1, 2):
                x, y = point
                x = max(0, min(x, self.field_width))
                y = max(0, min(y, self.field_height))
                result.append((float(x), float(y)))

            return result

        except Exception as e:
            print(f"Error transforming points: {e}")
            return [self._fallback_transform(point) for point in pixel_points]

    def inverse_transform_point(
        self, field_point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Transform a real-world field coordinate to pixel coordinate.

        Args:
            field_point: (x, y) coordinates in meters on the field

        Returns:
            (x, y) coordinates in pixels
        """
        if self.inverse_perspective_matrix is None:
            return self._fallback_inverse_transform(field_point)

        try:
            # Convert point to homogeneous coordinates
            point = np.array([[field_point[0], field_point[1]]], dtype=np.float32)

            # Apply inverse perspective transformation
            transformed = cv2.perspectiveTransform(
                point.reshape(1, 1, 2), self.inverse_perspective_matrix
            )

            # Extract coordinates
            x, y = transformed[0, 0]

            return (float(x), float(y))

        except Exception as e:
            print(f"Error inverse transforming point: {e}")
            return self._fallback_inverse_transform(field_point)

    def calculate_distance(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        """
        Calculate real-world distance between two field coordinates.

        Args:
            point1: First point (x, y) in meters
            point2: Second point (x, y) in meters

        Returns:
            Distance in meters
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return float(np.sqrt(dx * dx + dy * dy))

    def calculate_speed(
        self, point1: Tuple[float, float], point2: Tuple[float, float], time_diff: float
    ) -> float:
        """
        Calculate speed between two field coordinates.

        Args:
            point1: First point (x, y) in meters
            point2: Second point (x, y) in meters
            time_diff: Time difference in seconds

        Returns:
            Speed in km/h
        """
        if time_diff <= 0:
            return 0.0

        distance = self.calculate_distance(point1, point2)
        speed_ms = distance / time_diff  # meters per second
        speed_kmh = speed_ms * 3.6  # convert to km/h

        return speed_kmh

    def is_point_in_field(self, field_point: Tuple[float, float]) -> bool:
        """
        Check if a field coordinate is within the field boundaries.

        Args:
            field_point: (x, y) coordinates in meters

        Returns:
            True if point is within field, False otherwise
        """
        x, y = field_point
        return 0 <= x <= self.field_width and 0 <= y <= self.field_height

    def get_field_zone(self, field_point: Tuple[float, float]) -> str:
        """
        Get the field zone for a given coordinate.

        Args:
            field_point: (x, y) coordinates in meters

        Returns:
            Zone name as string
        """
        x, y = field_point

        # Divide field into zones
        third_width = self.field_width / 3
        third_height = self.field_height / 3

        # Determine horizontal zone
        if x < third_width:
            h_zone = "left"
        elif x < 2 * third_width:
            h_zone = "center"
        else:
            h_zone = "right"

        # Determine vertical zone
        if y < third_height:
            v_zone = "defensive"
        elif y < 2 * third_height:
            v_zone = "midfield"
        else:
            v_zone = "attacking"

        return f"{v_zone}_{h_zone}"

    def _fallback_transform(
        self, pixel_point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Fallback transformation when not calibrated."""
        # Simple linear scaling as fallback
        # Assumes video is roughly 1920x1080 and maps to field dimensions
        x_scale = self.field_width / 1920.0
        y_scale = self.field_height / 1080.0

        x = pixel_point[0] * x_scale
        y = pixel_point[1] * y_scale

        # Clamp to field boundaries
        x = max(0, min(x, self.field_width))
        y = max(0, min(y, self.field_height))

        return (x, y)

    def _fallback_inverse_transform(
        self, field_point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Fallback inverse transformation when not calibrated."""
        # Simple inverse linear scaling
        x_scale = 1920.0 / self.field_width
        y_scale = 1080.0 / self.field_height

        x = field_point[0] * x_scale
        y = field_point[1] * y_scale

        return (x, y)

    def get_calibration_info(self) -> Dict[str, Any]:
        """Get calibration information."""
        return {
            "is_calibrated": self.perspective_matrix is not None,
            "field_width": self.field_width,
            "field_height": self.field_height,
            "pixel_vertices": self.pixel_vertices,
            "field_vertices": self.field_vertices,
        }

    def reset(self):
        """Reset the transformer state."""
        self.perspective_matrix = None
        self.inverse_perspective_matrix = None
        self.pixel_vertices = None
