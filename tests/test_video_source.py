import unittest
from football_ai.sources.video.video_loader import VideoLoader
from football_ai.sources.video.video_analysis_processor import VideoAnalysisProcessor
from football_ai.sources.video.video_pipeline import VideoPipeline
from football_ai.sources.video.video_analysis_factory import create_demo_processor
from football_ai.core_models.video import Video
import os

# Use a real video file from input_videos if available
VIDEO_PATH = "input_videos/08fd33_4.mp4"
MODEL_PATH = "models/detect/yolo11n.pt"  # Example model path, adjust as needed


class TestVideoLoader(unittest.TestCase):
    def test_load_video(self):
        if not os.path.exists(VIDEO_PATH):
            self.skipTest(f"Test video not found: {VIDEO_PATH}")
        loader = VideoLoader(log_level="ERROR")
        video = loader.load_video(VIDEO_PATH, max_frames=5)
        self.assertIsInstance(video, Video)
        self.assertGreater(len(video.frames), 0)
        self.assertEqual(video.video_path, VIDEO_PATH)
        self.assertEqual(video.metadata.resolution[0] > 0, True)


class TestVideoAnalysisProcessor(unittest.TestCase):
    def test_processor_init(self):
        processor = VideoAnalysisProcessor(
            model_path=MODEL_PATH,
            confidence_threshold=0.3,
            device="cpu",
            log_level="ERROR",
        )
        self.assertEqual(processor.model_path, MODEL_PATH)
        self.assertEqual(processor.confidence_threshold, 0.3)


class TestVideoPipeline(unittest.TestCase):
    def test_pipeline_init(self):
        pipeline = VideoPipeline(
            model_path=MODEL_PATH,
            confidence_threshold=0.3,
            device="cpu",
            log_level="ERROR",
        )
        self.assertEqual(pipeline.model_path, MODEL_PATH)
        self.assertEqual(pipeline.confidence_threshold, 0.3)


class TestVideoAnalysisFactory(unittest.TestCase):
    def test_create_demo_processor(self):
        processor = create_demo_processor(
            MODEL_PATH, VIDEO_PATH, "outputs/demo_out.mp4"
        )
        self.assertIsInstance(processor, VideoAnalysisProcessor)
        self.assertEqual(processor.model_path, MODEL_PATH)


if __name__ == "__main__":
    unittest.main()
