from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class Field:
    """
    Comprehensive field model that consolidates all field-related data and functionality.
    """

    # Field identification
    field_id: Optional[str] = None
    field_name: Optional[str] = None
    field_type: str = "football"  # "football", "soccer", "futsal"

    # Field dimensions (in meters)
    length: float = 105.0  # Standard football field length
    width: float = 68.0  # Standard football field width

    # Goal dimensions (in meters)
    goal_width: float = 7.32
    goal_height: float = 2.44
    goal_depth: float = 2.0

    # Area dimensions (in meters)
    penalty_area_length: float = 16.5  # From goal line
    penalty_area_width: float = 40.32
    goal_area_length: float = 5.5  # From goal line (6-yard box)
    goal_area_width: float = 18.32
    center_circle_radius: float = 9.15

    # Field corners and boundaries
    field_corners: Optional[List[Tuple[float, float]]] = (
        None  # Pixel coordinates of field corners
    )
    field_boundaries: Optional[List[Tuple[float, float]]] = None  # All boundary points

    # Goal coordinates (pixel coordinates)
    left_goal_posts: Optional[List[Tuple[float, float]]] = (
        None  # [top_left, bottom_left, top_right, bottom_right]
    )
    right_goal_posts: Optional[List[Tuple[float, float]]] = None

    # Key field lines (pixel coordinates)
    center_line: Optional[List[Tuple[float, float]]] = None
    left_penalty_area: Optional[List[Tuple[float, float]]] = None
    right_penalty_area: Optional[List[Tuple[float, float]]] = None
    left_goal_area: Optional[List[Tuple[float, float]]] = None
    right_goal_area: Optional[List[Tuple[float, float]]] = None
    center_circle: Optional[Tuple[float, float, float]] = (
        None  # (center_x, center_y, radius)
    )

    # Transformation matrices
    perspective_matrix: Optional[np.ndarray] = None  # Homography matrix
    inverse_perspective_matrix: Optional[np.ndarray] = None

    # Calibration data
    calibration_confidence: float = 0.0
    is_calibrated: bool = False
    calibration_method: Optional[str] = (
        None  # "manual", "automatic", "template_matching"
    )
    calibration_timestamp: Optional[float] = None

    # Camera and view information
    camera_height: Optional[float] = None  # Camera height in meters
    camera_angle: Optional[float] = None  # Camera angle in degrees
    view_coverage: Optional[float] = None  # Percentage of field visible (0-100)

    # Field zones for tactical analysis
    zones: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    # Field surface and conditions
    surface_type: str = "grass"  # "grass", "artificial", "dirt"
    surface_condition: Optional[str] = None  # "dry", "wet", "muddy"

    # Match context
    stadium_name: Optional[str] = None
    match_date: Optional[str] = None
    weather_conditions: Optional[str] = None

    # Additional custom data
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default zones and validate field dimensions."""
        if not self.zones:
            self._initialize_default_zones()
        self._validate_dimensions()

    def _initialize_default_zones(self) -> None:
        """Initialize default tactical zones."""
        # Defensive, middle, and attacking thirds
        third_length = self.length / 3

        self.zones = {
            "left_defensive_third": [
                (0, 0),
                (third_length, 0),
                (third_length, self.width),
                (0, self.width),
            ],
            "middle_third": [
                (third_length, 0),
                (2 * third_length, 0),
                (2 * third_length, self.width),
                (third_length, self.width),
            ],
            "right_attacking_third": [
                (2 * third_length, 0),
                (self.length, 0),
                (self.length, self.width),
                (2 * third_length, self.width),
            ],
            # Additional zones can be added
            "center_zone": [
                (third_length, self.width / 4),
                (2 * third_length, self.width / 4),
                (2 * third_length, 3 * self.width / 4),
                (third_length, 3 * self.width / 4),
            ],
        }

    def _validate_dimensions(self) -> None:
        """Validate field dimensions against FIFA standards."""
        # FIFA minimum/maximum dimensions
        if not (90 <= self.length <= 120):
            print(f"Warning: Field length {self.length}m outside FIFA range (90-120m)")
        if not (45 <= self.width <= 90):
            print(f"Warning: Field width {self.width}m outside FIFA range (45-90m)")

    def set_calibration(
        self,
        corners: List[Tuple[float, float]],
        confidence: float,
        method: str = "manual",
    ) -> None:
        """Set field calibration data."""
        self.field_corners = corners
        self.calibration_confidence = confidence
        self.calibration_method = method
        self.is_calibrated = confidence > 0.7  # Threshold for good calibration

        if len(corners) == 4:
            self._calculate_perspective_matrix()

    def _calculate_perspective_matrix(self) -> None:
        """Calculate perspective transformation matrix from field corners."""
        if self.field_corners and len(self.field_corners) == 4:
            # Real-world field corners (in field coordinates)
            field_coords = np.array(
                [
                    [0, 0],  # Top-left
                    [self.length, 0],  # Top-right
                    [self.length, self.width],  # Bottom-right
                    [0, self.width],  # Bottom-left
                ],
                dtype=np.float32,
            )

            # Pixel coordinates
            pixel_coords = np.array(self.field_corners, dtype=np.float32)

            # Calculate homography matrix
            try:
                import cv2

                self.perspective_matrix = cv2.getPerspectiveTransform(
                    pixel_coords, field_coords
                )
                self.inverse_perspective_matrix = cv2.getPerspectiveTransform(
                    field_coords, pixel_coords
                )
            except ImportError:
                print("OpenCV not available - perspective matrix calculation skipped")

    def pixel_to_field(
        self, pixel_pos: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Convert pixel coordinates to field coordinates."""
        if self.perspective_matrix is None:
            return None

        try:
            import cv2

            pixel_point = np.array([[pixel_pos]], dtype=np.float32)
            field_point = cv2.perspectiveTransform(pixel_point, self.perspective_matrix)
            return (float(field_point[0][0][0]), float(field_point[0][0][1]))
        except ImportError:
            return None

    def field_to_pixel(
        self, field_pos: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Convert field coordinates to pixel coordinates."""
        if self.inverse_perspective_matrix is None:
            return None

        try:
            import cv2

            field_point = np.array([[field_pos]], dtype=np.float32)
            pixel_point = cv2.perspectiveTransform(
                field_point, self.inverse_perspective_matrix
            )
            return (float(pixel_point[0][0][0]), float(pixel_point[0][0][1]))
        except ImportError:
            return None

    def get_zone_for_position(self, field_pos: Tuple[float, float]) -> Optional[str]:
        """Determine which tactical zone a field position belongs to."""
        x, y = field_pos

        for zone_name, zone_coords in self.zones.items():
            if self._point_in_polygon(field_pos, zone_coords):
                return zone_name
        return None

    def _point_in_polygon(
        self, point: Tuple[float, float], polygon: List[Tuple[float, float]]
    ) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def is_in_penalty_area(
        self, field_pos: Tuple[float, float], side: str = "both"
    ) -> bool:
        """Check if position is in penalty area."""
        x, y = field_pos

        # Left penalty area
        left_penalty = (
            0 <= x <= self.penalty_area_length
            and (self.width - self.penalty_area_width) / 2
            <= y
            <= (self.width + self.penalty_area_width) / 2
        )

        # Right penalty area
        right_penalty = (
            self.length - self.penalty_area_length <= x <= self.length
            and (self.width - self.penalty_area_width) / 2
            <= y
            <= (self.width + self.penalty_area_width) / 2
        )

        if side == "left":
            return left_penalty
        elif side == "right":
            return right_penalty
        else:
            return left_penalty or right_penalty

    def is_in_goal_area(
        self, field_pos: Tuple[float, float], side: str = "both"
    ) -> bool:
        """Check if position is in goal area (6-yard box)."""
        x, y = field_pos

        # Left goal area
        left_goal = (
            0 <= x <= self.goal_area_length
            and (self.width - self.goal_area_width) / 2
            <= y
            <= (self.width + self.goal_area_width) / 2
        )

        # Right goal area
        right_goal = (
            self.length - self.goal_area_length <= x <= self.length
            and (self.width - self.goal_area_width) / 2
            <= y
            <= (self.width + self.goal_area_width) / 2
        )

        if side == "left":
            return left_goal
        elif side == "right":
            return right_goal
        else:
            return left_goal or right_goal

    def calculate_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Calculate distance between two field positions in meters."""
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def get_field_info(self) -> Dict[str, Union[str, float, bool]]:
        """Get comprehensive field information."""
        return {
            "field_id": self.field_id,
            "field_name": self.field_name,
            "field_type": self.field_type,
            "dimensions": f"{self.length}m x {self.width}m",
            "surface_type": self.surface_type,
            "is_calibrated": self.is_calibrated,
            "calibration_confidence": self.calibration_confidence,
            "calibration_method": self.calibration_method,
            "view_coverage": self.view_coverage,
            "stadium_name": self.stadium_name,
        }

    def get_tactical_zones(self) -> List[str]:
        """Get list of defined tactical zones."""
        return list(self.zones.keys())

    def add_custom_zone(
        self, zone_name: str, coordinates: List[Tuple[float, float]]
    ) -> None:
        """Add a custom tactical zone."""
        self.zones[zone_name] = coordinates

    def get_center_coordinates(self) -> Tuple[float, float]:
        """Get field center coordinates."""
        return (self.length / 2, self.width / 2)
