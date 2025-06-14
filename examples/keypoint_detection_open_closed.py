"""
Example: Future Keypoint Detection Extensibility

This example shows how coordinate transformer can be extended
with auto detection in the future.
"""

from football_ai import FootballAnalysisPipeline
from football_ai.domain.interfaces import FieldKeypointDetector
from football_ai.config import get_default_config
from typing import List, Optional
import numpy as np


def example_basic_usage():
    """Example: Basic usage with different strategies."""

    # Method 1: Manual field corners (current approach)
    pipeline = FootballAnalysisPipeline()
    pipeline.set_field_keypoints(
        [[100.0, 100.0], [900.0, 100.0], [900.0, 600.0], [100.0, 600.0]]
    )
    print(f"Current strategy: {pipeline.get_current_keypoint_strategy()}")

    # Method 2: Switch to automatic detection
    pipeline.switch_to_auto_detection("line_detection")
    print(f"Current strategy: {pipeline.get_current_keypoint_strategy()}")

    # Method 3: Switch to hybrid (auto + manual fallback)
    pipeline.switch_to_hybrid_strategy("deep_learning")
    print(f"Current strategy: {pipeline.get_current_keypoint_strategy()}")


def example_configuration_based():
    """Example: Configuration-based strategy selection."""

    # Configure strategy at initialization
    config = get_default_config()
    config.transformation.keypoint_strategy = "hybrid"
    config.transformation.auto_detection_algorithm = "line_detection"
    config.transformation.default_keypoints = [
        [100.0, 100.0],
        [900.0, 100.0],
        [900.0, 600.0],
        [100.0, 600.0],
    ]

    pipeline = FootballAnalysisPipeline(config=config)
    print(f"Initialized with strategy: {pipeline.get_current_keypoint_strategy()}")


def example_adding_new_algorithm():
    """
    Example: How to add a new detection algorithm without modifying existing code.

    This demonstrates the Open-Closed Principle in action.
    """

    # Step 1: Create new detection algorithm (NEW CODE, doesn't modify existing)
    class YOLOKeypointDetector(FieldKeypointDetector):
        """New algorithm using YOLO for keypoint detection."""

        def __init__(self):
            self._last_confidence = 0.0

        def detect_keypoints(
            self, frame: np.ndarray, **kwargs
        ) -> Optional[List[List[float]]]:
            # Your YOLO-based detection logic here
            print("Running YOLO keypoint detection...")
            # Return detected keypoints or None
            return [
                [120.0, 120.0],
                [880.0, 120.0],
                [880.0, 580.0],
                [120.0, 580.0],
            ]  # Mock result

        def get_confidence(self) -> float:
            return 0.95  # Mock confidence

        def get_algorithm_name(self) -> str:
            return "YOLO Keypoints"

        def is_ready(self) -> bool:
            return True

    # Step 2: Future - Use new algorithm (when auto detection is implemented)
    pipeline = FootballAnalysisPipeline()

    # Create custom detector (future implementation)
    yolo_detector = YOLOKeypointDetector()

    # Future: Set auto detector when implemented
    # pipeline.coordinate_transformer.set_auto_detector(yolo_detector)
    # success = pipeline.coordinate_transformer.auto_calibrate_from_frame(frame)
    print("Auto detection not yet implemented - use manual field corners for now")

    print(f"Using new algorithm: {pipeline.get_current_keypoint_strategy()}")


def example_runtime_switching():
    """Example: Runtime switching between different strategies."""

    pipeline = FootballAnalysisPipeline()

    print("=== Runtime Strategy Switching Demo ===")

    # Start with manual field corners
    manual_keypoints = [[100.0, 100.0], [900.0, 100.0], [900.0, 600.0], [100.0, 600.0]]
    pipeline.set_field_keypoints(manual_keypoints)
    print(f"1. {pipeline.get_current_keypoint_strategy()}")

    # Future: Switch to auto detection when implemented
    # pipeline.coordinate_transformer.set_auto_detector(detector)
    print("2. Auto detection not yet implemented")
    print(f"2. {pipeline.get_current_keypoint_strategy()}")

    # Switch to deep learning
    pipeline.switch_to_auto_detection("deep_learning")
    print(f"3. {pipeline.get_current_keypoint_strategy()}")

    # Switch to hybrid with template matching
    pipeline.switch_to_hybrid_strategy("template_matching")
    print(f"4. {pipeline.get_current_keypoint_strategy()}")

    # Back to manual (using config defaults)
    pipeline.set_field_keypoints()  # Uses config defaults
    print(f"5. {pipeline.get_current_keypoint_strategy()}")


if __name__ == "__main__":
    print("=== Open-Closed Principle Keypoint Detection Examples ===\n")

    print("1. Basic Usage:")
    example_basic_usage()
    print()

    print("2. Configuration-Based:")
    example_configuration_based()
    print()

    print("3. Adding New Algorithm (Open-Closed Principle):")
    example_adding_new_algorithm()
    print()

    print("4. Runtime Switching:")
    example_runtime_switching()
