#!/usr/bin/env python3
"""
Refactoring verification test.

This script tests that all our refactoring changes work correctly and
that the pipeline still functions as expected.
"""

import sys
import os

sys.path.insert(0, "/workspaces/football_analysis")


def test_imports():
    """Test that all imports work correctly."""
    print("🔍 Testing imports...")

    try:
        # Test main pipeline import
        from football_ai import FootballAnalysisPipeline, get_default_config

        print("✅ Main pipeline imports successful")

        # Test utilities
        from football_ai.utils import (
            setup_logger,
            validate_video_path,
            create_progress_bar,
        )

        print("✅ Utilities imports successful")

        # Test common imports through utils
        from football_ai.utils import np, cv2, torch, logging

        print("✅ Common imports through utils successful")

        # Test processors
        from football_ai.detection.object_detection_processor import (
            ObjectDetectionProcessor,
        )
        from football_ai.tracking.track_processor import TrackProcessor
        from football_ai.assignment.team_assignment_processor import (
            SigLIPTeamAssignmentProcessor,
        )

        print("✅ Processor imports successful")

        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\n🔧 Testing utility functions...")

    try:
        from football_ai.utils import setup_logger, create_progress_bar

        # Test logger
        logger = setup_logger("test_logger")
        logger.info("Test log message")
        print("✅ Logger utility works")

        # Test progress bar (without actually running it)
        progress_bar = create_progress_bar(total=100, desc="Test", disable=True)
        progress_bar.close()
        print("✅ Progress bar utility works")

        return True
    except Exception as e:
        print(f"❌ Utilities error: {e}")
        return False


def test_pipeline_initialization():
    """Test pipeline initialization."""
    print("\n🚀 Testing pipeline initialization...")

    try:
        from football_ai import FootballAnalysisPipeline, get_default_config

        config = get_default_config()
        config.strict_mode = False  # For testing

        pipeline = FootballAnalysisPipeline(config=config)

        print(f"✅ Pipeline initialized with {len(pipeline.processors)} processors")

        # Test processor types
        processor_types = [type(p).__name__ for p in pipeline.processors]
        print(f"📋 Processors: {', '.join(processor_types)}")

        return True
    except Exception as e:
        print(f"❌ Pipeline initialization error: {e}")
        return False


def test_config_system():
    """Test configuration system."""
    print("\n⚙️ Testing configuration system...")

    try:
        from football_ai import (
            get_default_config,
            get_high_accuracy_config,
            get_fast_processing_config,
        )

        # Test different configs
        default_config = get_default_config()
        high_accuracy_config = get_high_accuracy_config()
        fast_config = get_fast_processing_config()

        print("✅ All configuration presets load successfully")
        print(f"📊 Default config processors: {len(default_config.get_summary())}")

        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_backwards_compatibility():
    """Test that our changes maintain backwards compatibility."""
    print("\n🔄 Testing backwards compatibility...")

    try:
        # Test that old imports still work
        from football_ai.domain.interfaces import Processor
        from football_ai.domain.data_models import VideoData, FrameData

        print("✅ Core interfaces still accessible")

        # Test that processors still inherit from the right base
        from football_ai.detection.object_detection_processor import (
            ObjectDetectionProcessor,
        )

        # Test class inheritance without initializing (avoid file path issues)
        assert issubclass(ObjectDetectionProcessor, Processor)
        print("✅ Processors still implement Processor interface")

        return True
    except Exception as e:
        print(f"❌ Backwards compatibility error: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 REFACTORING VERIFICATION TESTS")
    print("=" * 50)

    tests = [
        test_imports,
        test_utilities,
        test_pipeline_initialization,
        test_config_system,
        test_backwards_compatibility,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n❌ Test {test.__name__} failed!")

    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Refactoring successful!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
