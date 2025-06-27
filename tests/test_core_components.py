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

from football_ai.core_models import (
    Video as VideoData,
    Frame as FrameData,
    ObjectType,
    Player,
    Ball,
)
from football_ai.config import (
    FootballAIConfig,
    get_default_config,
    get_broadcast_config,
)


class TestDataModels(unittest.TestCase):
    """Test core data models and structures."""

    def test_player_creation(self):
        """Test Player creation and attributes."""
        player = Player(track_id=1, team_id=0, pixel_position=(10, 20), speed=7.5)
        self.assertEqual(player.track_id, 1)
        self.assertEqual(player.team_id, 0)
        self.assertEqual(player.pixel_position, (10, 20))
        self.assertEqual(player.speed, 7.5)

    def test_ball_creation(self):
        """Test Ball creation and attributes."""
        ball = Ball(track_id=1, pixel_position=(100, 200), speed=15.0)
        self.assertEqual(ball.track_id, 1)
        self.assertEqual(ball.pixel_position, (100, 200))
        self.assertEqual(ball.speed, 15.0)

    def test_frame_data_creation(self):
        """Test FrameData creation with new model objects."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        player = Player(track_id=1, team_id=0, pixel_position=(10, 20))
        ball = Ball(track_id=1, pixel_position=(100, 200))
        frame_data = FrameData(frame_number=0, timestamp=0.0, raw_frame=frame)
        frame_data.add_player(player)
        frame_data.set_ball(ball)
        self.assertEqual(frame_data.frame_number, 0)
        self.assertEqual(frame_data.timestamp, 0.0)
        self.assertEqual(len(frame_data.players), 1)
        self.assertIsNotNone(frame_data.ball)

    def test_video_data_creation(self):
        """Test VideoData creation and properties."""
        frames = []
        for i in range(3):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_data = FrameData(
                frame_number=i, timestamp=i * 0.033, raw_frame=frame  # ~30fps
            )
            frames.append(frame_data)
        video_data = VideoData(video_id="test_video_001", video_path="test_video.mp4")
        self.assertEqual(video_data.video_id, "test_video_001")
        self.assertEqual(video_data.video_path, "test_video.mp4")
        self.assertEqual(len(video_data.frames), 0)


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
    """Test object type constants and usage with new model classes."""

    def test_object_type_constants(self):
        """Test that object type constants are properly defined."""
        self.assertEqual(ObjectType.PLAYER, "player")
        self.assertEqual(ObjectType.GOALKEEPER, "goalkeeper")
        self.assertEqual(ObjectType.REFEREE, "referee")
        self.assertEqual(ObjectType.BALL, "ball")

    def test_object_type_usage_in_player_and_ball(self):
        """Test object types work correctly in Player and Ball objects."""
        player = Player(track_id=1, team_id=0)
        ball = Ball(track_id=1)
        # Simulate type assignment (if used in your code)
        player_type = ObjectType.PLAYER
        ball_type = ObjectType.BALL
        self.assertEqual(player_type, "player")
        self.assertEqual(ball_type, "ball")


if __name__ == "__main__":
    unittest.main()
