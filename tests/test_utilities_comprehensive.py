"""
Comprehensive Unit Tests for Football Analysis Utilities

Tests all utility classes including ConfigFactory, VideoUtils, LoggingUtils, etc.
"""

import unittest
import tempfile
import os
import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import cv2
import numpy as np

# Import utility classes
from football_ai.utilities.config_factory import ConfigFactory
from football_ai.utilities.video_utils import (
    validate_video_path,
    get_video_info,
    create_video_writer,
)
from football_ai.utilities.logging_utils import setup_logger
from football_ai.utilities.progress_utils import create_progress_bar
from football_ai.config import FootballAIConfig


class TestConfigFactory(unittest.TestCase):
    """Test ConfigFactory utility class."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = ConfigFactory()

    def test_create_test_config(self):
        """Test creating test configuration."""
        config = ConfigFactory.create_test_config()

        self.assertIsInstance(config, FootballAIConfig)
        self.assertFalse(config.show_progress_bars)

    def test_create_test_config_with_params(self):
        """Test creating test configuration with parameters."""
        config = ConfigFactory.create_test_config(
            model_path="/test/model.pt",
            input_video_path="/test/input.mp4",
            output_video_path="/test/output.mp4",
            strict_mode=True,
        )

        self.assertIsInstance(config, FootballAIConfig)
        self.assertTrue(config.strict_mode)
        self.assertFalse(config.show_progress_bars)

    def test_create_demo_config(self):
        """Test creating demo configuration."""
        config = ConfigFactory.create_demo_config()

        self.assertIsInstance(config, FootballAIConfig)

    def test_create_evaluation_config(self):
        """Test creating evaluation configuration."""
        config = ConfigFactory.create_evaluation_config("/test/input.mp4")

        self.assertIsInstance(config, FootballAIConfig)
        self.assertTrue(config.debug_mode)

    def test_create_demo_config_with_params(self):
        """Test creating demo configuration with custom parameters."""
        config = ConfigFactory.create_demo_config(
            model_path="/custom/model.pt",
            input_video_path="/custom/input.mp4",
            output_video_path="/custom/output.mp4",
        )

        self.assertIsInstance(config, FootballAIConfig)
        self.assertTrue(config.show_progress_bars)


class TestVideoUtils(unittest.TestCase):
    """Test VideoUtils utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary video file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = os.path.join(self.temp_dir, "test_video.mp4")

        # Create a dummy video file (just an empty file for path testing)
        with open(self.test_video_path, "w") as f:
            f.write("")

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_video_path):
            os.remove(self.test_video_path)
        os.rmdir(self.temp_dir)

    def test_validate_video_path_valid(self):
        """Test video path validation with valid path."""
        # Mock cv2.VideoCapture to return a valid capture
        with patch("cv2.VideoCapture") as mock_cap:
            mock_cap_instance = Mock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap.return_value = mock_cap_instance

            result = validate_video_path(self.test_video_path)
            self.assertTrue(result)

    def test_validate_video_path_invalid(self):
        """Test video path validation with invalid path."""
        result = validate_video_path("/nonexistent/path.mp4")
        self.assertFalse(result)

    def test_validate_video_path_unreadable(self):
        """Test video path validation with unreadable file."""
        with patch("cv2.VideoCapture") as mock_cap:
            mock_cap_instance = Mock()
            mock_cap_instance.isOpened.return_value = False
            mock_cap.return_value = mock_cap_instance

            result = validate_video_path(self.test_video_path)
            self.assertFalse(result)

    def test_get_video_info_file_not_found(self):
        """Test get_video_info with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            get_video_info("/nonexistent/path.mp4")

    def test_get_video_info_valid_file(self):
        """Test get_video_info with valid file."""
        with patch("cv2.VideoCapture") as mock_cap:
            mock_cap_instance = Mock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 1800,
            }.get(prop, 0)
            mock_cap.return_value = mock_cap_instance

            width, height, fps, frame_count = get_video_info(self.test_video_path)

            self.assertEqual(width, 1920)
            self.assertEqual(height, 1080)
            self.assertEqual(fps, 30.0)
            self.assertEqual(frame_count, 1800)

    def test_get_video_info_invalid_file(self):
        """Test get_video_info with invalid file."""
        with patch("cv2.VideoCapture") as mock_cap:
            mock_cap_instance = Mock()
            mock_cap_instance.isOpened.return_value = False
            mock_cap.return_value = mock_cap_instance

            with self.assertRaises(ValueError):
                get_video_info(self.test_video_path)

    def test_create_video_writer_valid(self):
        """Test creating video writer with valid parameters."""
        output_path = os.path.join(self.temp_dir, "output.mp4")

        with patch("cv2.VideoWriter") as mock_writer:
            mock_writer_instance = Mock()
            mock_writer_instance.isOpened.return_value = True
            mock_writer.return_value = mock_writer_instance

            writer = create_video_writer(output_path, 1920, 1080, 30.0)

            self.assertIsNotNone(writer)
            mock_writer.assert_called_once()

    def test_create_video_writer_invalid(self):
        """Test creating video writer with invalid parameters."""
        with patch("cv2.VideoWriter") as mock_writer:
            mock_writer_instance = Mock()
            mock_writer_instance.isOpened.return_value = False
            mock_writer.return_value = mock_writer_instance

            with self.assertRaises(ValueError):
                create_video_writer("/invalid/path.mp4", 1920, 1080, 30.0)


class TestLoggingUtils(unittest.TestCase):
    """Test LoggingUtils utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear any existing loggers
        logging.getLogger().handlers.clear()

    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        logger = setup_logger("test_logger")

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_logger")
        self.assertEqual(logger.level, logging.INFO)

    def test_setup_logger_with_level(self):
        """Test logger setup with specific level."""
        logger = setup_logger("test_logger_debug", "DEBUG")

        self.assertEqual(logger.level, logging.DEBUG)

    def test_setup_logger_with_file(self):
        """Test logger setup with file output."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        temp_file.close()

        try:
            logger = setup_logger("test_logger_file", "INFO", log_file=temp_file.name)

            self.assertIsInstance(logger, logging.Logger)
            logger.info("Test message")

            # Check if file was created and contains log message
            self.assertTrue(os.path.exists(temp_file.name))

        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def test_setup_logger_invalid_level(self):
        """Test logger setup with invalid level."""
        # Should handle invalid level gracefully, defaulting to INFO
        logger = setup_logger("test_logger_invalid", "INVALID_LEVEL")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.level, logging.INFO)

    def test_setup_logger_duplicate_name(self):
        """Test logger setup with duplicate name."""
        logger1 = setup_logger("duplicate_logger")
        logger2 = setup_logger("duplicate_logger")

        # Should return the same logger instance
        self.assertEqual(logger1, logger2)


class TestProgressUtils(unittest.TestCase):
    """Test ProgressUtils utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_create_progress_bar_basic(self):
        """Test basic progress bar creation."""
        iterable = range(10)
        progress_bar = create_progress_bar(iterable=iterable, desc="Testing")

        # Test that we can iterate through it
        results = list(progress_bar)
        self.assertEqual(len(results), 10)
        self.assertEqual(results, list(range(10)))

    def test_create_progress_bar_with_unit(self):
        """Test progress bar creation with unit."""
        iterable = range(5)
        progress_bar = create_progress_bar(
            iterable=iterable, desc="Testing", unit="items"
        )

        results = list(progress_bar)
        self.assertEqual(len(results), 5)

    def test_create_progress_bar_empty_iterable(self):
        """Test progress bar with empty iterable."""
        iterable = []
        progress_bar = create_progress_bar(iterable=iterable, desc="Empty test")

        results = list(progress_bar)
        self.assertEqual(len(results), 0)

    def test_create_progress_bar_with_total(self):
        """Test progress bar with explicit total."""
        progress_bar = create_progress_bar(total=3, desc="Test")

        # Test with manual updates
        self.assertIsNotNone(progress_bar)
        progress_bar.close()

    def test_create_progress_bar_disabled(self):
        """Test progress bar creation when disabled."""
        iterable = range(5)
        progress_bar = create_progress_bar(iterable=iterable, desc="Test", disable=True)

        results = list(progress_bar)
        self.assertEqual(len(results), 5)


class TestPitchUtils(unittest.TestCase):
    """Test PitchUtils utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_pitch_utils_imports(self):
        """Test that pitch utils can be imported."""
        try:
            import football_ai.utilities.pitch_utils

            self.assertTrue(True)
        except ImportError as e:
            # If import fails due to missing dependencies, skip
            self.skipTest(f"PitchUtils import failed: {e}")

    def test_pitch_coordinate_conversion(self):
        """Test pitch coordinate conversion functions."""
        # This is a placeholder test - actual implementation would test
        # pixel to field coordinate conversion functions
        pass

    def test_pitch_area_detection(self):
        """Test pitch area detection functions."""
        # This is a placeholder test - actual implementation would test
        # penalty area, goal area detection functions
        pass


class TestUtilitiesIntegration(unittest.TestCase):
    """Test utilities integration and workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_and_logging_integration(self):
        """Test integration between config factory and logging."""
        config = ConfigFactory.create_test_config()
        logger = setup_logger("integration_test")

        self.assertIsInstance(config, FootballAIConfig)
        self.assertIsInstance(logger, logging.Logger)

    def test_video_and_progress_integration(self):
        """Test integration between video utils and progress bars."""
        # Mock video processing with progress bar
        frames = range(10)
        progress_bar = create_progress_bar(iterable=frames, desc="Processing frames")

        processed_frames = []
        for frame in progress_bar:
            processed_frames.append(frame)

        self.assertEqual(len(processed_frames), 10)

    def test_utilities_error_handling(self):
        """Test error handling across utilities."""
        # Test that utilities handle errors gracefully
        with self.assertRaises(FileNotFoundError):
            get_video_info("/nonexistent/file.mp4")

        # Test that logger handles invalid level gracefully
        logger = setup_logger("test", "INVALID_LEVEL")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.level, logging.INFO)


class TestUtilitiesPerformance(unittest.TestCase):
    """Test utilities performance with larger datasets."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_progress_bar_large_dataset(self):
        """Test progress bar with large dataset."""
        large_iterable = range(1000)
        progress_bar = create_progress_bar(
            iterable=large_iterable, desc="Large test", disable=True
        )

        # Test that it can handle large datasets without issues
        count = 0
        for item in progress_bar:
            count += 1
            if count > 100:  # Don't process all items in test
                break

        self.assertGreater(count, 100)

    def test_logging_performance(self):
        """Test logging performance."""
        logger = setup_logger("performance_test")

        # Test that logging doesn't significantly impact performance
        import time

        start_time = time.time()

        for i in range(100):
            logger.debug(f"Debug message {i}")

        end_time = time.time()
        elapsed = end_time - start_time

        # Should complete quickly (less than 1 second for 100 log messages)
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    unittest.main()
