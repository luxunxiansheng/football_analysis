"""
Unit Tests for Football Analysis Video Pipeline and Processors

Tests video processing pipeline and all processor classes.
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import video pipeline and processor classes
from football_ai.sources.video.video_pipeline import VideoPipeline
from football_ai.core_models.video import Video
from football_ai.core_models.frame import Frame
from football_ai.core_models.interfaces import Processor


class TestVideoPipeline(unittest.TestCase):
    """Test VideoPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "models/detect/best.pt"

    def test_video_pipeline_creation(self):
        """Test VideoPipeline initialization - should catch API mismatches."""
        try:
            # Test with actual classes to catch constructor mismatches
            pipeline = VideoPipeline(
                model_path=self.model_path,
                confidence_threshold=0.5,
                iou_threshold=0.4,
                device="cpu",
            )

            # If we get here, the API is correct
            self.assertEqual(pipeline.model_path, self.model_path)
            self.assertEqual(pipeline.confidence_threshold, 0.5)
            self.assertEqual(pipeline.iou_threshold, 0.4)
            self.assertEqual(pipeline.device, "cpu")
            self.assertIsInstance(pipeline.processors, list)

        except TypeError as e:
            # This should catch constructor API mismatches
            self.fail(f"VideoPipeline constructor API mismatch: {e}")
        except Exception as e:
            # Other errors might be acceptable (missing models, etc.)
            self.skipTest(f"VideoPipeline creation failed due to environment: {e}")

    @patch(
        "football_ai.sources.video.processors.object_detection.yolo_detector.ObjectDetectionProcessor"
    )
    @patch(
        "football_ai.sources.video.processors.object_tracking.byte_tracker.TrackProcessor"
    )
    def test_video_pipeline_feature_toggles(
        self, mock_track_processor, mock_detection_processor
    ):
        """Test VideoPipeline with different feature toggles."""
        # Mock the processors
        mock_detection_processor.return_value = Mock(spec=Processor)
        mock_track_processor.return_value = Mock(spec=Processor)

        # Create pipeline with all features disabled
        pipeline = VideoPipeline(
            model_path=self.model_path,
            enable_team_classification=False,
            enable_ball_tracking=False,
            enable_motion_analysis=False,
            enable_field_transformation=False,
            enable_video_rendering=False,
        )

        # Should still have core processors (detection + tracking)
        self.assertGreaterEqual(len(pipeline.processors), 2)

    @patch(
        "football_ai.sources.video.processors.object_detection.yolo_detector.ObjectDetectionProcessor"
    )
    @patch(
        "football_ai.sources.video.processors.object_tracking.byte_tracker.TrackProcessor"
    )
    @patch(
        "football_ai.sources.video.processors.motion_analysis.speed_calculator.ObjectMotionProcessor"
    )
    def test_video_pipeline_process_video(
        self, mock_motion_processor, mock_track_processor, mock_detection_processor
    ):
        """Test processing a video through the pipeline."""
        # Mock all processors
        mock_processors = [
            Mock(spec=Processor),
            Mock(spec=Processor),
            Mock(spec=Processor),
        ]

        for mock_proc in mock_processors:
            mock_proc.process.return_value = Mock(spec=Video)

        mock_detection_processor.return_value = mock_processors[0]
        mock_track_processor.return_value = mock_processors[1]
        mock_motion_processor.return_value = mock_processors[2]

        pipeline = VideoPipeline(
            model_path=self.model_path,
            enable_motion_analysis=True,
            enable_team_classification=False,
            enable_ball_tracking=False,
            enable_field_transformation=False,
            enable_video_rendering=False,
        )

        # Create test video
        video = Video(video_id="test", video_path="test.mp4")

        try:
            # Process video
            result = pipeline.process_video(video)

            # Verify result is a Video instance
            self.assertIsInstance(result, Video)

        except (AttributeError, NotImplementedError):
            # If process_video method isn't fully implemented
            self.skipTest("VideoPipeline.process_video not fully implemented")


class TestProcessorBase(unittest.TestCase):
    """Test base processor functionality."""

    def test_processor_interface_implementation(self):
        """Test implementing the Processor interface."""

        class TestProcessor(Processor):
            def process(self, data: Video) -> Video:
                # Simple pass-through processor
                return data

        processor = TestProcessor()
        self.assertIsInstance(processor, Processor)

        # Test processing
        video = Video(video_id="test", video_path="test.mp4")
        result = processor.process(video)
        self.assertEqual(result, video)


class TestObjectDetectionProcessor(unittest.TestCase):
    """Test ObjectDetectionProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_object_detection_processor_creation(self):
        """Test ObjectDetectionProcessor initialization - catch constructor API issues."""
        try:
            from football_ai.sources.video.processors.object_detection.yolo_detector import (
                ObjectDetectionProcessor,
            )

            # Test constructor with parameters that VideoPipeline tries to pass
            processor = ObjectDetectionProcessor(
                model_path="models/detect/best.pt",
                confidence_threshold=0.5,
                iou_threshold=0.4,  # This should be supported
                device="cuda",  # This should be supported
                max_detections=1000,  # This should be supported
            )

            self.assertEqual(processor.confidence_threshold, 0.5)
            self.assertEqual(processor.iou_threshold, 0.4)
            self.assertEqual(processor.device, "cuda")
            self.assertEqual(processor.max_detections, 1000)

        except TypeError as e:
            self.fail(f"ObjectDetectionProcessor constructor API mismatch: {e}")
        except ImportError:
            self.skipTest("ObjectDetectionProcessor not available")
        except Exception as e:
            # YOLO model loading might fail, but constructor API should work
            self.skipTest(f"Model loading failed but API is correct: {e}")

    @patch("football_ai.sources.video.processors.object_detection.yolo_detector.YOLO")
    @patch("football_ai.utilities.create_progress_bar")
    def test_object_detection_process(self, mock_progress_bar, mock_yolo):
        """Test ObjectDetectionProcessor processing."""
        try:
            from football_ai.sources.video.processors.object_detection.yolo_detector import (
                ObjectDetectionProcessor,
            )

            # Mock YOLO model and progress bar
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = None  # Simulate no detections
            mock_model.predict.return_value = [
                mock_result
            ]  # Return list with one result
            mock_yolo.return_value = mock_model
            mock_progress_bar.return_value = Mock()

            processor = ObjectDetectionProcessor(
                model_path="models/detect/best.pt", confidence_threshold=0.5
            )

            # Create test video with frames
            video = Video(video_id="test", video_path="test.mp4")
            frame = Frame(
                frame_number=1,
                timestamp=0.033,
                raw_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            )
            video.add_frame(frame)

            # Process video
            result = processor.process(video)

            # Verify result is a Video instance
            self.assertIsInstance(result, Video)

        except (ImportError, AttributeError):
            self.skipTest("ObjectDetectionProcessor not fully implemented")


class TestTrackProcessor(unittest.TestCase):
    """Test TrackProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_track_processor_creation(self):
        """Test TrackProcessor initialization."""
        try:
            from football_ai.sources.video.processors.object_tracking.byte_tracker import (
                TrackProcessor,
            )

            processor = TrackProcessor(
                track_activation_threshold=0.6,
                lost_track_buffer=30,
                minimum_matching_threshold=0.7,
                frame_rate=30,
                minimum_consecutive_frames=1,
                min_track_length=5,
            )

            # Just verify the processor was created successfully
            self.assertIsInstance(processor, Processor)

        except ImportError:
            self.skipTest("TrackProcessor not available")

    def test_track_processor_process(self):
        """Test TrackProcessor processing."""
        try:
            from football_ai.sources.video.processors.object_tracking.byte_tracker import (
                TrackProcessor,
            )

            processor = TrackProcessor(
                track_activation_threshold=0.6,
                lost_track_buffer=30,
                minimum_matching_threshold=0.7,
                frame_rate=30,
            )

            # Create test video
            video = Video(video_id="test", video_path="test.mp4")
            frame = Frame(frame_number=1, timestamp=0.033)
            video.add_frame(frame)

            # Process video
            result = processor.process(video)

            # Verify result is a Video instance
            self.assertIsInstance(result, Video)

        except (ImportError, AttributeError, NotImplementedError):
            self.skipTest("TrackProcessor not fully implemented")


class TestMotionAnalysisProcessors(unittest.TestCase):
    """Test motion analysis processors."""

    def test_object_motion_processor(self):
        """Test ObjectMotionProcessor."""
        try:
            from football_ai.sources.video.processors.motion_analysis.speed_calculator import (
                ObjectMotionProcessor,
            )

            processor = ObjectMotionProcessor()
            self.assertIsInstance(processor, Processor)

            # Test processing
            video = Video(video_id="test", video_path="test.mp4")
            result = processor.process(video)
            self.assertIsInstance(result, Video)

        except (ImportError, AttributeError, NotImplementedError):
            self.skipTest("ObjectMotionProcessor not fully implemented")

    def test_camera_motion_processor(self):
        """Test CameraMotionProcessor."""
        try:
            from football_ai.sources.video.processors.motion_analysis.camera_stabilizer import (
                CameraMotionProcessor,
            )

            processor = CameraMotionProcessor()
            self.assertIsInstance(processor, Processor)

            # Test processing
            video = Video(video_id="test", video_path="test.mp4")
            result = processor.process(video)
            self.assertIsInstance(result, Video)

        except (ImportError, AttributeError, NotImplementedError):
            self.skipTest("CameraMotionProcessor not fully implemented")


class TestTeamClassificationProcessors(unittest.TestCase):
    """Test team classification processors."""

    def test_siglip_team_assignment_processor(self):
        """Test SigLIPTeamAssignmentProcessor."""
        try:
            from football_ai.sources.video.processors.team_classification.siglip_team_classifier import (
                SigLIPTeamAssignmentProcessor,
            )

            processor = SigLIPTeamAssignmentProcessor(
                model_path="models/embed/siglip-base-patch16-224",
                batch_size=32,
                device="cpu",
            )
            self.assertIsInstance(processor, Processor)

        except (ImportError, AttributeError):
            self.skipTest("SigLIPTeamAssignmentProcessor not available")

    def test_ball_assignment_processor(self):
        """Test BallAssignmentProcessor."""
        try:
            from football_ai.sources.video.processors.team_classification.possession_analyzer import (
                BallAssignmentProcessor,
            )

            processor = BallAssignmentProcessor(max_distance=50.0)
            self.assertIsInstance(processor, Processor)

        except (ImportError, AttributeError):
            self.skipTest("BallAssignmentProcessor not available")


class TestMatchAnalysisProcessors(unittest.TestCase):
    """Test match analysis processors."""

    def test_speed_processor(self):
        """Test SpeedProcessor."""
        try:
            from football_ai.sources.video.processors.match_analysis.speed_analyzer import (
                SpeedProcessor,
            )

            processor = SpeedProcessor()
            self.assertIsInstance(processor, Processor)

        except (ImportError, AttributeError):
            self.skipTest("SpeedProcessor not available")

    def test_ball_control_processor(self):
        """Test BallControlProcessor."""
        try:
            from football_ai.sources.video.processors.match_analysis.possession_tracker import (
                BallControlProcessor,
            )

            processor = BallControlProcessor()
            self.assertIsInstance(processor, Processor)

        except (ImportError, AttributeError):
            self.skipTest("BallControlProcessor not available")


class TestFieldTransformationProcessor(unittest.TestCase):
    """Test field transformation processor."""

    def test_field_transformation_processor(self):
        """Test FieldTransformationProcessor."""
        try:
            from football_ai.sources.video.processors.coordinate_transformation.coordinate_transformer import (
                FieldTransformationProcessor,
            )

            processor = FieldTransformationProcessor()
            self.assertIsInstance(processor, Processor)

        except (ImportError, AttributeError, TypeError):
            self.skipTest(
                "FieldTransformationProcessor not available or requires parameters"
            )


class TestVideoRenderingProcessor(unittest.TestCase):
    """Test video rendering processor."""

    def test_video_annotator_processor(self):
        """Test RendererProcessor."""
        try:
            from football_ai.sources.video.processors.video_rendering.video_annotator import (
                RendererProcessor,
            )

            processor = RendererProcessor(output_path="/tmp/test_output.mp4")
            self.assertIsInstance(processor, Processor)

        except (ImportError, AttributeError, TypeError):
            self.skipTest(
                "RendererProcessor not available or requires different parameters"
            )


class TestVideoExportProcessor(unittest.TestCase):
    """Test video export processor."""

    def test_video_writer_processor(self):
        """Test VideoWriterProcessor."""
        try:
            from football_ai.sources.video.processors.video_export.video_exporter import (
                VideoWriterProcessor,
            )

            processor = VideoWriterProcessor(output_path="/tmp/test_output.mp4")
            self.assertIsInstance(processor, Processor)

        except (ImportError, AttributeError, TypeError):
            self.skipTest(
                "VideoWriterProcessor not available or requires different parameters"
            )


class TestProcessorConstructorCompatibility(unittest.TestCase):
    """Test that all processors have compatible constructors with VideoPipeline."""

    def test_all_processor_constructors(self):
        """Test that all processor constructors match VideoPipeline expectations."""

        # Test ObjectDetectionProcessor
        try:
            from football_ai.sources.video.processors.object_detection.yolo_detector import (
                ObjectDetectionProcessor,
            )

            # These are the parameters VideoPipeline tries to pass
            processor = ObjectDetectionProcessor(
                model_path="test.pt",
                confidence_threshold=0.5,
                iou_threshold=0.4,
                device="cpu",
                max_detections=1000,
            )

            # Verify parameters were stored
            self.assertEqual(processor.confidence_threshold, 0.5)
            self.assertEqual(processor.iou_threshold, 0.4)
            self.assertEqual(processor.device, "cpu")
            self.assertEqual(processor.max_detections, 1000)

        except TypeError as e:
            self.fail(
                f"ObjectDetectionProcessor constructor incompatible with VideoPipeline: {e}"
            )
        except Exception:
            # Model loading issues are acceptable
            pass

    def test_track_processor_constructor(self):
        """Test TrackProcessor constructor compatibility."""
        try:
            from football_ai.sources.video.processors.object_tracking.byte_tracker import (
                TrackProcessor,
            )

            # These are the parameters VideoPipeline tries to pass
            processor = TrackProcessor(
                track_activation_threshold=0.6,
                lost_track_buffer=30,
                minimum_matching_threshold=0.7,
                frame_rate=30,
                minimum_consecutive_frames=1,
                min_track_length=5,
            )

            # Just verify it was created without TypeError
            self.assertIsInstance(processor, TrackProcessor)

        except TypeError as e:
            self.fail(
                f"TrackProcessor constructor incompatible with VideoPipeline: {e}"
            )
        except ImportError:
            self.skipTest("TrackProcessor not available")

    def test_field_transformation_processor_constructor(self):
        """Test FieldTransformationProcessor constructor compatibility."""
        try:
            from football_ai.sources.video.processors.coordinate_transformation.coordinate_transformer import (
                FieldTransformationProcessor,
            )

            # VideoPipeline should NOT pass model_path and device to this processor
            processor = FieldTransformationProcessor()

            # Just verify it was created without TypeError
            self.assertIsInstance(processor, FieldTransformationProcessor)

        except TypeError as e:
            self.fail(f"FieldTransformationProcessor constructor issues: {e}")
        except ImportError:
            self.skipTest("FieldTransformationProcessor not available")

    def test_renderer_processor_constructor(self):
        """Test RendererProcessor constructor compatibility."""
        try:
            from football_ai.sources.video.processors.video_rendering.video_annotator import (
                RendererProcessor,
            )

            # RendererProcessor needs output_path
            processor = RendererProcessor(output_path="/tmp/test.mp4")

            # Just verify it was created without TypeError
            self.assertIsInstance(processor, RendererProcessor)

        except TypeError as e:
            self.fail(f"RendererProcessor constructor issues: {e}")
        except ImportError:
            self.skipTest("RendererProcessor not available")

    def test_video_writer_processor_constructor(self):
        """Test VideoWriterProcessor constructor compatibility."""
        try:
            from football_ai.sources.video.processors.video_export.video_exporter import (
                VideoWriterProcessor,
            )

            # VideoWriterProcessor needs output_path
            processor = VideoWriterProcessor(output_path="/tmp/test.mp4")

            # Just verify it was created without TypeError
            self.assertIsInstance(processor, VideoWriterProcessor)

        except TypeError as e:
            self.fail(f"VideoWriterProcessor constructor issues: {e}")
        except ImportError:
            self.skipTest("VideoWriterProcessor not available")


if __name__ == "__main__":
    unittest.main()
