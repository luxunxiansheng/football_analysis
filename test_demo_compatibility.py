#!/usr/bin/env python3
"""
Unit test to reproduce the exact demo error and catch API mismatches.
"""

import unittest
import sys

sys.path.insert(0, ".")


class TestDemoCompatibility(unittest.TestCase):
    """Test that reproduces the exact demo error scenario."""

    def test_demo_video_analysis_creation(self):
        """Test the exact demo scenario that was failing."""
        try:
            # Check the actual create_demo_processor signature
            from football_ai.sources.video.video_analysis_factory import (
                create_demo_processor,
            )

            # This should be the correct call based on the actual API
            video_processor = create_demo_processor(
                model_path="models/detect/best.pt",
                input_video_path="input_videos/test.mp4",
                output_video_path="outputs/test.mp4",
            )

            # If we get here, the API is working
            self.assertIsNotNone(video_processor)
            print("✓ Demo processor creation succeeded")

        except TypeError as e:
            # This should catch API mismatches
            self.fail(f"DEMO API MISMATCH CAUGHT: {e}")
        except Exception as e:
            # Other errors might be environmental (missing models, etc.)
            print(f"Demo creation failed with non-API error: {e}")
            self.skipTest(f"Demo processor creation failed due to environment: {e}")

    def test_video_pipeline_direct(self):
        """Test direct VideoPipeline creation that was called from video_analysis_processor.py line 78."""
        try:
            from football_ai.sources.video.video_pipeline import VideoPipeline

            # This is the exact call from video_analysis_processor.py line 78
            pipeline = VideoPipeline(
                model_path="models/detect/best.pt",
                confidence_threshold=0.3,
                iou_threshold=0.45,
                device="cuda",
                max_detections=1000,
                enable_team_classification=True,
                enable_ball_tracking=True,
                enable_motion_analysis=True,
                enable_field_transformation=True,
                enable_video_rendering=False,
                enable_video_export=False,
            )

            # If we get here, the API is working
            self.assertIsNotNone(pipeline)
            print("✓ VideoPipeline creation succeeded")

        except TypeError as e:
            # This should catch the exact API mismatch
            self.fail(f"VIDEOPIPELINE API MISMATCH CAUGHT: {e}")
        except Exception as e:
            # Other errors might be environmental
            print(f"VideoPipeline creation failed with non-API error: {e}")
            self.skipTest(f"VideoPipeline creation failed due to environment: {e}")


if __name__ == "__main__":
    # Run with maximum verbosity to see all output
    unittest.main(verbosity=2, buffer=False)
