"""
Test Runner for Football Analysis Project

Runs all unit tests for the football analysis system.
"""

import unittest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import all test modules
from tests.test_core_models import *
from tests.test_game_management import *
from tests.test_video_pipeline import *
from tests.test_utilities import *
from tests.test_analytics import *

# Also import the original test file
from tests.test_core_components import *


def create_test_suite():
    """Create a comprehensive test suite."""
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        # Core Models Tests
        TestPlayer,
        TestBall,
        TestField,
        TestFrame,
        TestVideo,
        TestGame,
        TestGoalkeeper,
        TestReferee,
        TestProcessorInterface,
        TestCameraMotion,
        TestProcessingStatus,
        TestVideoMetadata,
        TestMatchContext,
        # Game Management Tests
        TestGameManager,
        TestGameFactory,
        TestTeamOperations,
        TestMatchEvents,
        TestAnalysisSource,
        # Video Pipeline Tests
        TestVideoPipeline,
        TestProcessorBase,
        TestObjectDetectionProcessor,
        TestTrackProcessor,
        TestMotionAnalysisProcessors,
        TestTeamClassificationProcessors,
        TestMatchAnalysisProcessors,
        TestFieldTransformationProcessor,
        TestVideoRenderingProcessor,
        TestVideoExportProcessor,
        # Utilities Tests
        TestLoggingUtils,
        TestProgressUtils,
        TestVideoUtils,
        TestConfigFactory,
        TestPitchUtils,
        TestCommonImports,
        TestUtilitiesInit,
        # Analytics Tests
        TestMatchStatistics,
        TestHeatmapGenerator,
        TestAnalyticsIntegration,
        TestAnalyticsPerformance,
        # Original Core Components Tests
        TestDataModels,
        TestConfiguration,
        TestObjectTypes,
    ]

    # Add tests from each class
    for test_class in test_classes:
        try:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        except NameError:
            # Skip test classes that aren't available
            print(f"Warning: Test class {test_class.__name__} not available")
            continue

    return test_suite


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    print("=" * 60)
    print("FOOTBALL ANALYSIS PROJECT - COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    # Create and run test suite
    test_suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            error_msg = traceback.split("\n")[-2] if "\n" in traceback else traceback
            print(f"  - {test}: {error_msg}")

    if result.skipped:
        print(f"\nSKIPPED ({len(result.skipped)}):")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")

    success_rate = (
        (result.testsRun - len(result.failures) - len(result.errors))
        / result.testsRun
        * 100
        if result.testsRun > 0
        else 0
    )
    print(f"\nSUCCESS RATE: {success_rate:.1f}%")

    return result.wasSuccessful()


def run_specific_test_module(module_name):
    """Run tests from a specific module."""
    print(f"Running tests from {module_name}...")

    if module_name == "core_models":
        test_classes = [TestPlayer, TestBall, TestField, TestFrame, TestVideo, TestGame]
    elif module_name == "game_management":
        test_classes = [TestGameManager, TestGameFactory, TestTeamOperations]
    elif module_name == "video_pipeline":
        test_classes = [
            TestVideoPipeline,
            TestObjectDetectionProcessor,
            TestTrackProcessor,
        ]
    elif module_name == "utilities":
        test_classes = [TestLoggingUtils, TestProgressUtils, TestVideoUtils]
    elif module_name == "analytics":
        test_classes = [
            TestMatchStatistics,
            TestHeatmapGenerator,
            TestAnalyticsIntegration,
        ]
    else:
        print(f"Unknown module: {module_name}")
        return False

    test_suite = unittest.TestSuite()
    for test_class in test_classes:
        try:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        except NameError:
            continue

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Football Analysis Tests")
    parser.add_argument(
        "--module",
        choices=[
            "core_models",
            "game_management",
            "video_pipeline",
            "utilities",
            "analytics",
        ],
        help="Run tests for a specific module",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Test output verbosity (0=quiet, 1=normal, 2=verbose)",
    )

    args = parser.parse_args()

    if args.module:
        success = run_specific_test_module(args.module)
    else:
        success = run_tests(args.verbosity)

    sys.exit(0 if success else 1)
