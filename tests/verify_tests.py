#!/usr/bin/env python3
"""
Simple test verification script to check if our test framework works.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_basic_imports():
    """Test basic imports work."""
    print("Testing basic imports...")

    try:
        import football_ai

        print("✓ football_ai package imported successfully")
    except Exception as e:
        print(f"✗ football_ai package import failed: {e}")
        return False

    try:
        from football_ai.core_models.player import Player

        print("✓ Player class imported successfully")

        # Test creating a player
        player = Player(track_id=1)
        print("✓ Player instance created successfully")
        print(f"  Player track_id: {player.track_id}")

    except Exception as e:
        print(f"✗ Player import/creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    try:
        from football_ai.core_models.video import Video

        print("✓ Video class imported successfully")

        # Test creating a video
        video = Video(video_id="test", video_path="test.mp4")
        print("✓ Video instance created successfully")
        print(f"  Video ID: {video.video_id}")

    except Exception as e:
        print(f"✗ Video import/creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_simple_unittest():
    """Test that unittest framework works."""
    print("\nTesting unittest framework...")

    import unittest

    class SimpleTest(unittest.TestCase):
        def test_basic_assertion(self):
            self.assertEqual(1 + 1, 2)

        def test_player_creation(self):
            from football_ai.core_models.player import Player

            player = Player(track_id=42)
            self.assertEqual(player.track_id, 42)

    # Run the simple test
    suite = unittest.TestLoader().loadTestsFromTestCase(SimpleTest)
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("✓ Simple unittest execution successful")
        return True
    else:
        print("✗ Simple unittest execution failed")
        for failure in result.failures:
            print(f"  Failure: {failure[0]} - {failure[1]}")
        for error in result.errors:
            print(f"  Error: {error[0]} - {error[1]}")
        return False


def main():
    """Main test verification function."""
    print("=" * 60)
    print("FOOTBALL ANALYSIS PROJECT - TEST VERIFICATION")
    print("=" * 60)

    all_tests_passed = True

    # Test basic imports
    if not test_basic_imports():
        all_tests_passed = False

    # Test unittest framework
    if not test_simple_unittest():
        all_tests_passed = False

    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✓ ALL VERIFICATION TESTS PASSED")
        print("The test framework is ready to use!")
    else:
        print("✗ SOME VERIFICATION TESTS FAILED")
        print("Please check the errors above and fix dependencies.")
    print("=" * 60)

    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
