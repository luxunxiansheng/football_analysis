"""
Comprehensive test to verify the refactored codebase works correctly.
"""


def test_refactored_system():
    """Test that all refactored components work together."""

    print("🧪 Testing Refactored Football Analysis System")
    print("=" * 50)

    # Test 1: Import all utilities
    print("1. Testing utilities imports...")
    try:
        from football_ai.utils import (
            np,
            cv2,
            torch,
            logging,
            setup_logger,
            create_progress_bar,
            validate_video_path,
            get_video_info,
        )
        from football_ai.utils.config_factory import ConfigFactory

        print("   ✅ All utilities imported successfully")
    except Exception as e:
        print(f"   ❌ Utilities import failed: {e}")
        return False

    # Test 2: Test logging utility
    print("2. Testing logging utility...")
    try:
        logger = setup_logger("TestLogger", "INFO")
        logger.info("Test log message")
        print("   ✅ Logging utility works")
    except Exception as e:
        print(f"   ❌ Logging utility failed: {e}")
        return False

    # Test 3: Test progress bar utility
    print("3. Testing progress bar utility...")
    try:
        progress = create_progress_bar(
            total=5, desc="Test progress", unit="items", disable=True
        )
        for i in range(5):
            progress.update(1)
        progress.close()
        print("   ✅ Progress bar utility works")
    except Exception as e:
        print(f"   ❌ Progress bar utility failed: {e}")
        return False

    # Test 4: Test configuration factory
    print("4. Testing configuration factory...")
    try:
        config = ConfigFactory.create_test_config(strict_mode=False)
        demo_config = ConfigFactory.create_demo_config()
        eval_config = ConfigFactory.create_evaluation_config("test.mp4")
        print("   ✅ Configuration factory works")
    except Exception as e:
        print(f"   ❌ Configuration factory failed: {e}")
        return False

    # Test 5: Test pipeline initialization
    print("5. Testing pipeline initialization...")
    try:
        from football_ai import FootballAnalysisPipeline

        config = ConfigFactory.create_test_config(
            model_path="models/detect/best.pt", strict_mode=False
        )
        pipeline = FootballAnalysisPipeline(config=config)
        print(f"   ✅ Pipeline initialized with {len(pipeline.processors)} processors")
    except Exception as e:
        print(f"   ❌ Pipeline initialization failed: {e}")
        return False

    # Test 6: Test individual processor imports
    print("6. Testing processor imports...")
    try:
        from football_ai.detection.object_detection_processor import (
            ObjectDetectionProcessor,
        )
        from football_ai.tracking.track_processor import TrackProcessor
        from football_ai.assignment.ball_assignment_processor import (
            BallAssignmentProcessor,
        )
        from football_ai.assignment.team_assignment_processor import (
            SigLIPTeamAssignmentProcessor,
        )

        print("   ✅ All processors imported successfully")
    except Exception as e:
        print(f"   ❌ Processor imports failed: {e}")
        return False

    print("\n🎉 All refactoring tests passed!")
    print("✨ The codebase has been successfully refactored and is ready to use!")
    return True


if __name__ == "__main__":
    test_refactored_system()
