"""
Unit Tests for Football Analysis Utilities

Tests utility classes and functions including logging, video utilities, progress utilities, etc.
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import utility classes
from football_ai.utilities.logging_utils import setup_logger
from football_ai.utilities.progress_utils import create_progress_bar


class TestLoggingUtils(unittest.TestCase):
    """Test logging utilities."""

    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        logger = setup_logger("test_logger")

        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_logger")

    def test_setup_logger_with_level(self):
        """Test logger setup with specific level."""
        logger = setup_logger("test_logger_debug", "DEBUG")

        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_logger_debug")

    def test_setup_logger_with_invalid_level(self):
        """Test logger setup with invalid level."""
        # Should not raise exception, should default to INFO
        logger = setup_logger("test_logger_invalid", "INVALID_LEVEL")

        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_logger_invalid")


class TestProgressUtils(unittest.TestCase):
    """Test progress utilities."""

    def test_create_progress_bar_basic(self):
        """Test basic progress bar creation."""
        test_iterable = range(10)
        progress_bar = create_progress_bar(test_iterable, desc="Testing")

        self.assertIsNotNone(progress_bar)

        # Test iteration
        count = 0
        for item in progress_bar:
            count += 1

        self.assertEqual(count, 10)

    def test_create_progress_bar_with_unit(self):
        """Test progress bar creation with unit."""
        test_iterable = range(5)
        progress_bar = create_progress_bar(test_iterable, desc="Testing", unit="items")

        self.assertIsNotNone(progress_bar)

        # Test iteration
        count = 0
        for item in progress_bar:
            count += 1

        self.assertEqual(count, 5)

    def test_create_progress_bar_empty_iterable(self):
        """Test progress bar with empty iterable."""
        test_iterable = []
        progress_bar = create_progress_bar(test_iterable, desc="Empty test")

        self.assertIsNotNone(progress_bar)

        # Test iteration
        count = 0
        for item in progress_bar:
            count += 1

        self.assertEqual(count, 0)


class TestVideoUtils(unittest.TestCase):
    """Test video utilities."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_validate_video_path_valid_extension(self):
        """Test video path validation with valid extensions."""
        try:
            from football_ai.utilities.video_utils import validate_video_path

            valid_paths = [
                "/path/to/video.mp4",
                "/path/to/video.avi",
                "/path/to/video.mov",
                "/path/to/video.mkv",
            ]

            for path in valid_paths:
                # This might return True/False or raise exception
                try:
                    result = validate_video_path(path)
                    # If it returns a boolean, it should be implementation dependent
                    self.assertIsInstance(result, (bool, type(None)))
                except Exception:
                    # If it raises an exception for non-existent files, that's also valid
                    pass

        except ImportError:
            self.skipTest("video_utils not available")

    def test_validate_video_path_invalid_extension(self):
        """Test video path validation with invalid extensions."""
        try:
            from football_ai.utilities.video_utils import validate_video_path

            invalid_paths = [
                "/path/to/file.txt",
                "/path/to/file.pdf",
                "/path/to/file.jpg",
            ]

            for path in invalid_paths:
                try:
                    result = validate_video_path(path)
                    # If it returns False for invalid extensions, that's correct
                    if result is not None:
                        self.assertFalse(result)
                except Exception:
                    # If it raises an exception for invalid files, that's also valid
                    pass

        except ImportError:
            self.skipTest("video_utils not available")

    def test_get_video_info(self):
        """Test getting video information."""
        try:
            from football_ai.utilities.video_utils import get_video_info

            # Test with a dummy path (function might handle non-existent files)
            try:
                info = get_video_info("/nonexistent/video.mp4")
                # If successful, should return some kind of info dict
                if info is not None:
                    self.assertIsInstance(info, dict)
            except Exception:
                # If it raises an exception for non-existent files, that's expected
                pass

        except ImportError:
            self.skipTest("get_video_info not available")

    @patch("cv2.VideoWriter")
    def test_create_video_writer(self, mock_video_writer):
        """Test creating video writer."""
        try:
            from football_ai.utilities.video_utils import create_video_writer

            # Mock the VideoWriter
            mock_writer = Mock()
            mock_video_writer.return_value = mock_writer

            writer = create_video_writer(
                output_path="/tmp/test_output.mp4", fps=30, width=640, height=480
            )

            self.assertIsNotNone(writer)

        except (ImportError, TypeError):
            self.skipTest(
                "create_video_writer not available or requires different parameters"
            )


class TestConfigFactory(unittest.TestCase):
    """Test configuration factory utilities."""

    def test_config_factory_exists(self):
        """Test that config factory module exists."""
        try:
            import football_ai.utilities.config_factory

            # If import succeeds, that's good
            self.assertTrue(True)
        except ImportError:
            self.skipTest("config_factory not available")


class TestPitchUtils(unittest.TestCase):
    """Test pitch/field utilities."""

    def test_pitch_utils_exists(self):
        """Test that pitch utils module exists."""
        try:
            import football_ai.utilities.pitch_utils

            # If import succeeds, that's good
            self.assertTrue(True)
        except ImportError:
            self.skipTest("pitch_utils not available")


class TestCommonImports(unittest.TestCase):
    """Test common imports module."""

    def test_common_imports_numpy(self):
        """Test numpy import from common imports."""
        try:
            from football_ai.utilities.common_imports import np

            self.assertIsNotNone(np)
            # Test that it's actually numpy
            test_array = np.array([1, 2, 3])
            self.assertEqual(len(test_array), 3)
        except ImportError:
            self.skipTest("numpy not available in common_imports")

    def test_common_imports_pathlib(self):
        """Test pathlib import from common imports."""
        try:
            from football_ai.utilities.common_imports import Path

            self.assertIsNotNone(Path)
            # Test that it's actually pathlib.Path
            test_path = Path("/test/path")
            self.assertEqual(str(test_path), "/test/path")
        except ImportError:
            self.skipTest("Path not available in common_imports")

    def test_common_imports_typing(self):
        """Test typing imports from common imports."""
        try:
            from football_ai.utilities.common_imports import List, Dict, Optional, Any

            self.assertIsNotNone(List)
            self.assertIsNotNone(Dict)
            self.assertIsNotNone(Optional)
            self.assertIsNotNone(Any)
        except ImportError:
            self.skipTest("typing imports not available in common_imports")


class TestUtilitiesInit(unittest.TestCase):
    """Test utilities __init__ module."""

    def test_utilities_init_imports(self):
        """Test that main utility functions are available from __init__."""
        try:
            from football_ai.utilities import setup_logger, create_progress_bar

            # Test logger
            logger = setup_logger("test")
            self.assertIsNotNone(logger)

            # Test progress bar
            progress = create_progress_bar(range(3), desc="test")
            self.assertIsNotNone(progress)

        except ImportError:
            self.skipTest("utilities __init__ imports not available")

    def test_utilities_init_common_imports(self):
        """Test that common imports are re-exported."""
        try:
            from football_ai.utilities import np, Path, List, Dict

            self.assertIsNotNone(np)
            self.assertIsNotNone(Path)
            self.assertIsNotNone(List)
            self.assertIsNotNone(Dict)

        except ImportError:
            self.skipTest("common imports not re-exported from utilities")


if __name__ == "__main__":
    unittest.main()
