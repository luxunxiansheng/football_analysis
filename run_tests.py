#!/usr/bin/env python3
"""
Football Analysis Project - Comprehensive Test Suite

This script runs all unit tests for the football analysis system.
"""

import sys
import os
import unittest
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Run comprehensive test suite."""
    print("=" * 80)
    print("FOOTBALL ANALYSIS PROJECT - COMPREHENSIVE UNIT TEST SUITE")
    print("=" * 80)
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print()

    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = project_root / "tests"

    print(f"Discovering tests in: {start_dir}")

    # Load all tests from tests directory
    suite = loader.discover(str(start_dir), pattern="test_*.py")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True, failfast=False)

    print("Running tests...")
    print("-" * 80)

    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    # Calculate success rate
    if result.testsRun > 0:
        success_rate = (
            (result.testsRun - len(result.failures) - len(result.errors))
            / result.testsRun
            * 100
        )
        print(f"Success rate: {success_rate:.1f}%")

    # Show failures
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"{i}. {test}")
            # Show just the assertion error, not full traceback
            lines = traceback.strip().split("\n")
            for line in lines:
                if "AssertionError" in line:
                    print(f"   {line.strip()}")
                    break

    # Show errors
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"{i}. {test}")
            # Show just the error message, not full traceback
            lines = traceback.strip().split("\n")
            error_line = lines[-1] if lines else "Unknown error"
            print(f"   {error_line.strip()}")

    # Show skipped tests
    if result.skipped:
        print(f"\nSKIPPED ({len(result.skipped)}):")
        skip_reasons = {}
        for test, reason in result.skipped:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        for reason, count in skip_reasons.items():
            print(f"  {reason}: {count} tests")

    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The football analysis system is working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Check the failures and errors above.")
        if len(result.skipped) > 0:
            print("Note: Skipped tests are expected for modules not fully implemented.")

    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
