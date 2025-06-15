"""
Coordinate Transformation Module

This module provides functionality to transform pixel coordinates to real-world
field coordinates using perspective transformation and homography.
Includes field corner detection and management.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union

from ..domain.interfaces import CoordinateTransformer, FieldKeypointDetector
from ..constants import FieldDimensions


class PerspectiveCoordinateTransformer(CoordinateTransformer):
    """
    Implementation of coordinate transformation using perspective
    transformation to map pixel coordinates to real football field coordinates.

    Includes encapsulated field corner management and auto-detection capability.
    """

    def __init__(
        self,
        field_width: Optional[float] = None,
        field_height: Optional[float] = None,
        field_preset: Optional[str] = None,
        field_corners: Optional[List[List[float]]] = None,
        auto_detector: Optional[FieldKeypointDetector] = None,
    ):
        """
        Initialize the coordinate transformer.

        Args:
            field_width: Width of football field in meters (overrides preset)
            field_height: Height of football field in meters (overrides preset)
            field_preset: Predefined field dimensions (e.g., 'FIFA_STANDARD', 'YOUTH_U12')
                         Available presets: FIFA_STANDARD, FIFA_MINIMUM, FIFA_MAXIMUM,
                         PREMIER_LEAGUE, LA_LIGA, BUNDESLIGA, SERIE_A, MLS,
                         YOUTH_U12, YOUTH_U14, YOUTH_U16, SEVEN_A_SIDE, FIVE_A_SIDE, etc.
            field_corners: Optional 4 field corner coordinates in pixels
            auto_detector: Optional detector for automatic field corner detection
        """
        # Determine field dimensions
        if field_width is not None and field_height is not None:
            # Explicit dimensions provided
            self.field_width = field_width
            self.field_height = field_height
            self.field_preset_used = None
        elif field_preset is not None:
            # Use preset dimensions
            try:
                self.field_width, self.field_height = FieldDimensions.get_dimensions(
                    field_preset
                )
                self.field_preset_used = field_preset
            except ValueError as e:
                print(f"Warning: {e}")
                print("Available presets:")
                FieldDimensions.list_presets()
                print("Falling back to FIFA standard dimensions.")
                self.field_width, self.field_height = FieldDimensions.get_dimensions(
                    "FIFA_STANDARD"
                )
                self.field_preset_used = "FIFA_STANDARD"
        else:
            # Default to FIFA standard
            self.field_width, self.field_height = FieldDimensions.get_dimensions(
                "FIFA_STANDARD"
            )
            self.field_preset_used = "FIFA_STANDARD"

        # Transformation matrices
        self.perspective_matrix: Optional[np.ndarray] = None
        self.inverse_perspective_matrix: Optional[np.ndarray] = None

        # Field corners in pixel coordinates
        self._field_corners: Optional[List[List[float]]] = field_corners

        # Auto detection capability
        self._auto_detector: Optional[FieldKeypointDetector] = auto_detector

        # Default field vertices in real-world coordinates (meters)
        self.field_vertices = [
            [0, 0],  # Top-left corner
            [self.field_width, 0],  # Top-right corner
            [self.field_width, self.field_height],  # Bottom-right corner
            [0, self.field_height],  # Bottom-left corner
        ]

        # Auto-calibrate if field corners are provided
        if self._field_corners:
            self.calibrate_from_corners(self._field_corners)

    def set_field_corners(self, corners: List[List[float]]) -> bool:
        """
        Set field corners and automatically calibrate.

        Args:
            corners: List of [x, y] pixel coordinates for field corners
                    in order: [top-left, top-right, bottom-right, bottom-left]

        Returns:
            True if calibration successful, False otherwise
        """
        self._field_corners = corners
        return self.calibrate_from_corners(corners)

    def get_field_corners(self) -> Optional[List[List[float]]]:
        """Get current field corners."""
        return self._field_corners

    def set_auto_detector(self, detector: FieldKeypointDetector) -> None:
        """Set automatic field corner detector for future use."""
        self._auto_detector = detector

    def auto_calibrate_from_frame(self, frame: np.ndarray) -> bool:
        """
        Automatically detect field corners from video frame and calibrate.

        Args:
            frame: Video frame for field corner detection

        Returns:
            True if detection and calibration successful, False otherwise
        """
        if not self._auto_detector:
            print("No auto detector available for field corner detection")
            return False

        if not self._auto_detector.is_ready():
            print("Auto detector not ready")
            return False

        detected_corners = self._auto_detector.detect_keypoints(frame)

        if detected_corners is None:
            print("Failed to detect field corners from frame")
            return False

        print(
            f"Detected field corners with confidence: {self._auto_detector.get_confidence()}"
        )
        return self.set_field_corners(detected_corners)

    def is_calibrated(self) -> bool:
        """Check if transformer is calibrated and ready to use."""
        return self.perspective_matrix is not None

    def calibrate_from_corners(self, pixel_corners: List[List[float]]) -> bool:
        """
        Calibrate the transformer using field corner coordinates in pixels.

        Args:
            pixel_corners: List of [x, y] pixel coordinates for field corners
                          in order: [top-left, top-right, bottom-right, bottom-left]

        Returns:
            True if calibration successful, False otherwise
        """
        try:
            if len(pixel_corners) != 4:
                print("Error: Need exactly 4 corner points for calibration")
                return False

            self._field_corners = pixel_corners

            # Convert to numpy arrays
            src_points = np.array(pixel_corners, dtype=np.float32)
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

    def calibrate(self, pixel_vertices: List[List[float]]) -> bool:
        """
        Legacy method for backward compatibility.

        Args:
            pixel_vertices: List of [x, y] pixel coordinates for field corners

        Returns:
            True if calibration successful, False otherwise
        """
        return self.calibrate_from_corners(pixel_vertices)

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
            "field_preset_used": self.field_preset_used,
            "field_corners": self._field_corners,
            "field_vertices": self.field_vertices,
        }

    def reset(self):
        """Reset the transformer state."""
        self.perspective_matrix = None
        self.inverse_perspective_matrix = None
        self._field_corners = None

    # Convenience class methods for common field types
    @classmethod
    def for_fifa_standard(
        cls,
        field_corners: Optional[List[List[float]]] = None,
        auto_detector: Optional[FieldKeypointDetector] = None,
    ) -> "PerspectiveCoordinateTransformer":
        """Create transformer for FIFA standard field (68m x 105m)."""
        return cls(
            field_preset="FIFA_STANDARD",
            field_corners=field_corners,
            auto_detector=auto_detector,
        )

    @classmethod
    def for_youth(
        cls,
        age_group: str,
        field_corners: Optional[List[List[float]]] = None,
        auto_detector: Optional[FieldKeypointDetector] = None,
    ) -> "PerspectiveCoordinateTransformer":
        """
        Create transformer for youth field.

        Args:
            age_group: 'U12', 'U14', or 'U16'
            field_corners: Optional field corner coordinates
            auto_detector: Optional auto detector
        """
        preset_map = {"U12": "YOUTH_U12", "U14": "YOUTH_U14", "U16": "YOUTH_U16"}
        if age_group not in preset_map:
            raise ValueError(
                f"Unknown age group '{age_group}'. Available: {list(preset_map.keys())}"
            )

        return cls(
            field_preset=preset_map[age_group],
            field_corners=field_corners,
            auto_detector=auto_detector,
        )

    @classmethod
    def for_league(
        cls,
        league: str,
        field_corners: Optional[List[List[float]]] = None,
        auto_detector: Optional[FieldKeypointDetector] = None,
    ) -> "PerspectiveCoordinateTransformer":
        """
        Create transformer for specific league.

        Args:
            league: League name ('premier_league', 'la_liga', 'bundesliga', 'serie_a', 'mls', etc.)
            field_corners: Optional field corner coordinates
            auto_detector: Optional auto detector
        """
        league_map = {
            "premier_league": "PREMIER_LEAGUE",
            "la_liga": "LA_LIGA",
            "bundesliga": "BUNDESLIGA",
            "serie_a": "SERIE_A",
            "mls": "MLS",
            "uefa_champions": "UEFA_CHAMPIONS",
        }

        league_key = league.lower()
        if league_key not in league_map:
            raise ValueError(
                f"Unknown league '{league}'. Available: {list(league_map.keys())}"
            )

        return cls(
            field_preset=league_map[league_key],
            field_corners=field_corners,
            auto_detector=auto_detector,
        )

    @classmethod
    def for_small_sided(
        cls,
        game_type: str,
        field_corners: Optional[List[List[float]]] = None,
        auto_detector: Optional[FieldKeypointDetector] = None,
    ) -> "PerspectiveCoordinateTransformer":
        """
        Create transformer for small-sided games.

        Args:
            game_type: '5-a-side' or '7-a-side'
            field_corners: Optional field corner coordinates
            auto_detector: Optional auto detector
        """
        game_map = {"5-a-side": "FIVE_A_SIDE", "7-a-side": "SEVEN_A_SIDE"}
        if game_type not in game_map:
            raise ValueError(
                f"Unknown game type '{game_type}'. Available: {list(game_map.keys())}"
            )

        return cls(
            field_preset=game_map[game_type],
            field_corners=field_corners,
            auto_detector=auto_detector,
        )
