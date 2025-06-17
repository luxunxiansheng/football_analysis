"""
Integration Tests for Football Analysis Pipeline

Tests the main pipeline functionality and processor integration.
"""

import unittest
import tempfile
import os
import shutil
from pathlib import Path
import numpy as np
import cv2

from football_ai.pipeline import FootballAnalysisPipeline
from football_ai.config import get_default_config
from football_ai.domain.data_models import (
    VideoData,
    FrameData,
    Detection,
    BoundingBox,
    ObjectType,
)


class TestPipelineIntegration(unittest.TestCase):
    """Test complete pipeline integration."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.test_dir = tempfile.mkdtemp()
        cls.models_dir = os.path.join(cls.test_dir, "models")
        cls.videos_dir = os.path.join(cls.test_dir, "videos")
        cls.output_dir = os.path.join(cls.test_dir, "output")

        os.makedirs(cls.models_dir, exist_ok=True)
        os.makedirs(cls.videos_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)

        # Create a dummy model file
        cls.dummy_model_path = os.path.join(cls.models_dir, "dummy_model.pt")
        with open(cls.dummy_model_path, "w") as f:
            f.write("dummy model content")

        # Create a test video file
        cls.test_video_path = os.path.join(cls.videos_dir, "test_video.mp4")
        cls._create_test_video(cls.test_video_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)

    @classmethod
    def _create_test_video(cls, video_path: str):
        """Create a simple test video file."""
        # Create a simple test video with 30 frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))

        for i in range(30):
            # Create a frame with a moving rectangle
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            x = int(100 + i * 10)  # Moving rectangle
            cv2.rectangle(frame, (x, 200), (x + 50, 250), (0, 255, 0), -1)
            out.write(frame)

        out.release()

    def test_pipeline_initialization(self):
        """Test pipeline initialization with configuration."""
        config = get_default_config()
        config.update_paths(
            model_path=self.dummy_model_path,
            input_video_path=self.test_video_path,
            output_video_path=os.path.join(self.output_dir, "output.mp4"),
        )

        # Disable strict mode for testing with dummy models
        config.strict_mode = False

        pipeline = FootballAnalysisPipeline(config=config)

        self.assertIsNotNone(pipeline.config)
        self.assertIsNotNone(pipeline.logger)
        self.assertIsNotNone(pipeline.processors)
        self.assertGreater(len(pipeline.processors), 0)

    def test_video_loading(self):
        """Test video loading functionality."""
        config = get_default_config()
        config.strict_mode = False

        pipeline = FootballAnalysisPipeline(config=config)

        # Test successful video loading
        video_data = pipeline.load_video(self.test_video_path)

        self.assertIsInstance(video_data, VideoData)
        self.assertEqual(video_data.video_path, self.test_video_path)
        self.assertGreater(video_data.frame_rate, 0)
        self.assertGreater(len(video_data.frames), 0)
        self.assertEqual(video_data.resolution, (640, 480))

        # Test each frame
        for frame_data in video_data.frames:
            self.assertIsInstance(frame_data, FrameData)
            self.assertIsNotNone(frame_data.raw_frame)
            self.assertEqual(frame_data.raw_frame.shape, (480, 640, 3))

    def test_video_loading_nonexistent_file(self):
        """Test video loading with non-existent file."""
        config = get_default_config()
        pipeline = FootballAnalysisPipeline(config=config)

        with self.assertRaises(FileNotFoundError):
            pipeline.load_video("nonexistent_video.mp4")

    def test_analysis_summary_generation(self):
        """Test analysis summary generation."""
        # Create mock video data with detections
        frames = []
        for i in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Add some mock detections
            bbox1 = BoundingBox(x1=100 + i * 10, y1=200, x2=150 + i * 10, y2=250)
            detection1 = Detection(
                bbox=bbox1,
                object_type=ObjectType.PLAYER,
                metadata={"track_id": 1, "team": 1},
            )

            bbox2 = BoundingBox(x1=300, y1=100, x2=330, y2=130)
            detection2 = Detection(
                bbox=bbox2, object_type=ObjectType.BALL, metadata={"track_id": 2}
            )

            frame_data = FrameData(
                frame_number=i,
                timestamp=i * 0.033,
                raw_frame=frame,
                detections=[detection1, detection2],
            )
            frames.append(frame_data)

        video_data = VideoData(
            video_path=self.test_video_path,
            frame_rate=30.0,
            resolution=(640, 480),
            duration=0.167,
            frames=frames,
        )

        config = get_default_config()
        pipeline = FootballAnalysisPipeline(config=config)

        summary = pipeline.get_analysis_summary(video_data)

        # Verify summary structure
        self.assertIn("video_info", summary)
        self.assertIn("detection_summary", summary)
        self.assertIn("tracking_summary", summary)
        self.assertIn("team_summary", summary)

        # Verify video info
        video_info = summary["video_info"]
        self.assertEqual(video_info["frames_processed"], 5)
        self.assertEqual(video_info["resolution"], (640, 480))

        # Verify detection counts
        detection_summary = summary["detection_summary"]
        self.assertEqual(
            detection_summary["player"], 5
        )  # 1 player per frame * 5 frames
        self.assertEqual(detection_summary["ball"], 5)  # 1 ball per frame * 5 frames

        # Verify tracking info
        tracking_summary = summary["tracking_summary"]
        self.assertEqual(tracking_summary["unique_tracks"], 2)  # track_id 1 and 2

    def test_configuration_validation_in_pipeline(self):
        """Test that pipeline validates configuration properly."""
        config = get_default_config()

        # Set invalid configuration
        config.model.confidence_threshold = 1.5  # Invalid: > 1.0
        config.strict_mode = True

        pipeline = FootballAnalysisPipeline(config=config)

        # Should raise error when processing with invalid config in strict mode
        with self.assertRaises(ValueError):
            pipeline.process_video(self.test_video_path)


class TestProcessorChaining(unittest.TestCase):
    """Test that processors work together correctly."""

    def test_mock_processor_chain(self):
        """Test a mock processor chain with dummy data."""
        from football_ai.domain.interfaces import Processor

        class MockDetectionProcessor(Processor):
            def process(self, data: VideoData) -> VideoData:
                # Add mock detections to each frame
                for frame_data in data.frames:
                    bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
                    detection = Detection(bbox=bbox, object_type=ObjectType.PLAYER)
                    frame_data.detections = [detection]
                return data

        class MockTrackingProcessor(Processor):
            def process(self, data: VideoData) -> VideoData:
                # Add track IDs to detections
                track_id = 1
                for frame_data in data.frames:
                    if frame_data.detections:
                        for detection in frame_data.detections:
                            if detection.metadata is None:
                                detection.metadata = {}
                            detection.metadata["track_id"] = track_id
                            track_id += 1
                return data

        # Create test video data
        frames = []
        for i in range(3):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_data = FrameData(frame_number=i, timestamp=i * 0.033, raw_frame=frame)
            frames.append(frame_data)

        video_data = VideoData(
            video_path="test.mp4",
            frame_rate=30.0,
            resolution=(640, 480),
            duration=0.1,
            frames=frames,
        )

        # Process through chain
        detector = MockDetectionProcessor()
        tracker = MockTrackingProcessor()

        # Chain processors
        video_data = detector.process(video_data)
        video_data = tracker.process(video_data)

        # Verify results
        for frame_data in video_data.frames:
            self.assertIsNotNone(frame_data.detections)
            self.assertEqual(len(frame_data.detections), 1)

            detection = frame_data.detections[0]
            self.assertEqual(detection.object_type, ObjectType.PLAYER)
            self.assertIsNotNone(detection.metadata)
            self.assertIn("track_id", detection.metadata)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in pipeline."""

    def test_graceful_processor_failure(self):
        """Test pipeline continues when processor fails in non-strict mode."""
        from football_ai.domain.interfaces import Processor

        class FailingProcessor(Processor):
            def process(self, data: VideoData) -> VideoData:
                raise RuntimeError("Simulated processor failure")

        class WorkingProcessor(Processor):
            def process(self, data: VideoData) -> VideoData:
                # Add metadata to show this processor ran
                data.metadata["working_processor_ran"] = True
                return data

        config = get_default_config()
        config.strict_mode = False  # Allow graceful failure

        pipeline = FootballAnalysisPipeline(config=config)

        # Replace processors with test processors
        pipeline.processors = [FailingProcessor(), WorkingProcessor()]

        # Create dummy video data
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_data = FrameData(frame_number=0, timestamp=0.0, raw_frame=frame)
        video_data = VideoData(
            video_path="test.mp4",
            frame_rate=30.0,
            resolution=(640, 480),
            duration=0.033,
            frames=[frame_data],
        )

        # Should not raise error, but should log failure
        result = pipeline._process_through_pipeline(video_data)

        # Working processor should have run despite failing processor
        self.assertTrue(result.metadata.get("working_processor_ran", False))


if __name__ == "__main__":
    # Make sure cv2 is available for tests
    try:
        import cv2

        unittest.main()
    except ImportError:
        print("OpenCV not available, skipping integration tests")
        print("Install with: pip install opencv-python")
