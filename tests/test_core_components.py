"""
Unit Tests for Football Analysis Core Components

Tests core data models, configuration, and basic functionality.
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
import numpy as np

from football_ai.domain.data_models import (
    VideoData,
    FrameData,
    Detection,
    BoundingBox,
    ObjectType,
)
from football_ai.config import (
    FootballAIConfig,
    get_default_config,
    get_broadcast_config,
)


class TestDataModels(unittest.TestCase):
    """Test core data models and structures."""

    def test_bounding_box_creation(self):
        """Test BoundingBox creation and methods."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200, confidence=0.85)

        self.assertEqual(bbox.x1, 10)
        self.assertEqual(bbox.y1, 20)
        self.assertEqual(bbox.x2, 100)
        self.assertEqual(bbox.y2, 200)
        self.assertEqual(bbox.confidence, 0.85)

        # Test as_list method
        bbox_list = bbox.as_list()
        self.assertEqual(bbox_list, [10, 20, 100, 200])

    def test_detection_creation(self):
        """Test Detection creation with metadata."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detection = Detection(
            bbox=bbox,
            object_type=ObjectType.PLAYER,
            confidence=0.9,
            metadata={"track_id": 1, "team": 1},
        )

        self.assertEqual(detection.object_type, ObjectType.PLAYER)
        self.assertEqual(detection.confidence, 0.9)
        self.assertEqual(detection.metadata["track_id"], 1)
        self.assertEqual(detection.metadata["team"], 1)

    def test_detection_metadata_flexibility(self):
        """Test detection metadata can store various types of information."""
        bbox = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        detection = Detection(bbox=bbox)

        # Test metadata is initialized as empty dict
        self.assertIsNotNone(detection.metadata)
        self.assertIsInstance(detection.metadata, dict)
        self.assertEqual(len(detection.metadata), 0)

        # Test adding various metadata
        detection.metadata["track_id"] = 42
        detection.metadata["team"] = 2
        detection.metadata["speed"] = 15.5
        detection.metadata["field_position"] = (25.0, 40.0)
        detection.metadata["assigned_player"] = None

        self.assertEqual(detection.metadata["track_id"], 42)
        self.assertEqual(detection.metadata["team"], 2)
        self.assertEqual(detection.metadata["speed"], 15.5)
        self.assertEqual(detection.metadata["field_position"], (25.0, 40.0))
        self.assertIsNone(detection.metadata["assigned_player"])

    def test_frame_data_creation(self):
        """Test FrameData creation with detections."""
        # Create sample frame with detections
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        bbox1 = BoundingBox(x1=10, y1=20, x2=50, y2=80)
        detection1 = Detection(bbox=bbox1, object_type=ObjectType.PLAYER)

        bbox2 = BoundingBox(x1=100, y1=120, x2=140, y2=180)
        detection2 = Detection(bbox=bbox2, object_type=ObjectType.BALL)

        frame_data = FrameData(
            frame_number=0,
            timestamp=0.0,
            raw_frame=frame,
            detections=[detection1, detection2],
        )

        self.assertEqual(frame_data.frame_number, 0)
        self.assertEqual(frame_data.timestamp, 0.0)
        self.assertEqual(len(frame_data.detections), 2)
        self.assertEqual(frame_data.detections[0].object_type, ObjectType.PLAYER)
        self.assertEqual(frame_data.detections[1].object_type, ObjectType.BALL)

    def test_video_data_creation(self):
        """Test VideoData creation and properties."""
        # Create sample frames
        frames = []
        for i in range(3):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_data = FrameData(
                frame_number=i, timestamp=i * 0.033, raw_frame=frame  # ~30fps
            )
            frames.append(frame_data)

        video_data = VideoData(
            video_path="test_video.mp4",
            frame_rate=30.0,
            resolution=(640, 480),
            duration=0.1,
            frames=frames,
        )

        self.assertEqual(video_data.video_path, "test_video.mp4")
        self.assertEqual(video_data.frame_rate, 30.0)
        self.assertEqual(video_data.resolution, (640, 480))
        self.assertEqual(video_data.duration, 0.1)
        self.assertEqual(len(video_data.frames), 3)


class TestConfiguration(unittest.TestCase):
    """Test configuration system."""

    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = get_default_config()

        self.assertIsInstance(config, FootballAIConfig)
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.tracking)
        self.assertIsNotNone(config.rendering)

    def test_broadcast_config_creation(self):
        """Test broadcast configuration preset."""
        config = get_broadcast_config()

        self.assertIsInstance(config, FootballAIConfig)
        # Broadcast config should have higher quality settings
        self.assertGreaterEqual(config.model.confidence_threshold, 0.5)

    def test_config_path_updates(self):
        """Test configuration path update functionality."""
        config = get_default_config()

        config.update_paths(
            model_path="/test/model.pt",
            input_video_path="/test/input.mp4",
            output_video_path="/test/output.mp4",
        )

        self.assertEqual(config.model.player_model_path, "/test/model.pt")
        self.assertEqual(config.processing.input_video_path, "/test/input.mp4")
        self.assertEqual(config.processing.output_video_path, "/test/output.mp4")

    def test_config_validation(self):
        """Test configuration validation."""
        config = get_default_config()

        # Valid configuration should have no issues
        issues = config.validate()
        # Note: We expect some issues since test paths don't exist
        self.assertIsInstance(issues, list)

        # Test invalid confidence threshold
        config.model.confidence_threshold = 1.5  # Invalid: > 1.0
        issues = config.validate()
        self.assertTrue(
            any("confidence threshold" in issue.lower() for issue in issues)
        )

    def test_config_serialization(self):
        """Test configuration save/load functionality."""
        config = get_default_config()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config.save_to_file(f.name)

            # Load and verify
            loaded_config = FootballAIConfig.load_from_file(f.name)

            self.assertEqual(
                config.model.confidence_threshold,
                loaded_config.model.confidence_threshold,
            )
            self.assertEqual(
                config.tracking.track_threshold, loaded_config.tracking.track_threshold
            )

            # Cleanup
            os.unlink(f.name)

    def test_config_to_dict(self):
        """Test configuration dictionary conversion."""
        config = get_default_config()
        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertIn("model", config_dict)
        self.assertIn("tracking", config_dict)
        self.assertIn("rendering", config_dict)

        # Test reconstruction from dict
        new_config = FootballAIConfig.from_dict(config_dict)
        self.assertEqual(
            config.model.confidence_threshold, new_config.model.confidence_threshold
        )


class TestObjectTypes(unittest.TestCase):
    """Test object type constants."""

    def test_object_type_constants(self):
        """Test that object type constants are properly defined."""
        self.assertEqual(ObjectType.PLAYER, "player")
        self.assertEqual(ObjectType.GOALKEEPER, "goalkeeper")
        self.assertEqual(ObjectType.REFEREE, "referee")
        self.assertEqual(ObjectType.BALL, "ball")

    def test_object_type_usage_in_detection(self):
        """Test object types work correctly in Detection objects."""
        bbox = BoundingBox(x1=0, y1=0, x2=50, y2=50)

        player_detection = Detection(bbox=bbox, object_type=ObjectType.PLAYER)
        ball_detection = Detection(bbox=bbox, object_type=ObjectType.BALL)

        self.assertEqual(player_detection.object_type, "player")
        self.assertEqual(ball_detection.object_type, "ball")


if __name__ == "__main__":
    unittest.main()
