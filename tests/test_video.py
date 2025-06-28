import unittest
from football_ai.core_models.video import (
    Video,
    VideoMetadata,
    MatchContext,
    ProcessingConfig,
    Field,
    Frame,
)
from typing import Tuple, Dict, Any


class TestVideoMetadata(unittest.TestCase):
    def test_metadata_fields(self):
        meta = VideoMetadata(
            file_path="/tmp/test.mp4",
            file_size=123456,
            codec="h264",
            container_format="mp4",
            bitrate=1000,
            frame_rate=25.0,
            resolution=(1280, 720),
            duration=90.0,
            total_frames=2250,
            aspect_ratio=16 / 9,
            average_quality=0.95,
            quality_variance=0.01,
            compression_ratio=2.5,
        )
        self.assertEqual(meta.file_path, "/tmp/test.mp4")
        self.assertEqual(meta.file_size, 123456)
        self.assertEqual(meta.codec, "h264")
        self.assertEqual(meta.container_format, "mp4")
        self.assertEqual(meta.bitrate, 1000)
        self.assertEqual(meta.frame_rate, 25.0)
        self.assertEqual(meta.resolution, (1280, 720))
        self.assertEqual(meta.duration, 90.0)
        self.assertEqual(meta.total_frames, 2250)
        self.assertAlmostEqual(meta.aspect_ratio, 16 / 9)
        self.assertAlmostEqual(meta.average_quality, 0.95)
        self.assertAlmostEqual(meta.quality_variance, 0.01)
        self.assertAlmostEqual(meta.compression_ratio, 2.5)


class TestMatchContext(unittest.TestCase):
    def test_context_fields(self):
        ctx = MatchContext(
            match_id="M1",
            match_date="2025-06-27",
            competition="Test League",
            season="2025",
            round="Final",
            home_team="A",
            away_team="B",
            home_team_id="1",
            away_team_id="2",
            venue="Test Stadium",
            stadium_capacity=50000,
            referee="Ref A",
            weather="Sunny",
            temperature=25.5,
            attendance=48000,
            final_score=(2, 1),
            half_time_score=(1, 1),
            match_events=[{"type": "goal", "minute": 23}],
        )
        self.assertEqual(ctx.match_id, "M1")
        self.assertEqual(ctx.home_team, "A")
        self.assertEqual(ctx.away_team, "B")
        self.assertEqual(ctx.final_score, (2, 1))
        self.assertEqual(ctx.match_events[0]["type"], "goal")


class TestProcessingConfig(unittest.TestCase):
    def test_config_fields(self):
        cfg = ProcessingConfig(
            detection_model="yolo",
            detection_confidence=0.7,
            detection_iou=0.6,
            tracking_method="sort",
            tracking_max_age=40,
            tracking_min_hits=2,
            field_detection_method="hough",
            field_calibration_method="auto",
            analyze_formations=False,
            analyze_possession=False,
            analyze_movement=True,
            save_intermediate_results=True,
            output_format="csv",
            batch_size=4,
            num_workers=2,
            gpu_enabled=True,
        )
        self.assertEqual(cfg.detection_model, "yolo")
        self.assertEqual(cfg.detection_confidence, 0.7)
        self.assertEqual(cfg.tracking_method, "sort")
        self.assertTrue(cfg.gpu_enabled)
        self.assertEqual(cfg.output_format, "csv")


class TestVideo(unittest.TestCase):
    def setUp(self):
        self.meta = VideoMetadata(file_path="/tmp/test.mp4", frame_rate=30.0)
        self.ctx = MatchContext(match_id="M1", home_team="A", away_team="B")
        self.cfg = ProcessingConfig(detection_model="yolo")
        self.field = Field(length=105.0, width=68.0)
        self.frame1 = Frame(frame_number=1, timestamp=0.0)
        self.frame2 = Frame(frame_number=2, timestamp=0.04)
        self.video = Video(
            video_id="vid1",
            video_path="/tmp/test.mp4",
            metadata=self.meta,
            match_context=self.ctx,
            processing_config=self.cfg,
            frames=[self.frame1, self.frame2],
            global_field=self.field,
        )

    def test_video_fields(self):
        self.assertEqual(self.video.video_id, "vid1")
        self.assertEqual(self.video.metadata.file_path, "/tmp/test.mp4")
        self.assertEqual(self.video.match_context.home_team, "A")
        self.assertEqual(len(self.video.frames), 2)
        self.assertEqual(self.video.global_field.length, 105.0)

    def test_add_and_remove_frame(self):
        frame3 = Frame(frame_number=3, timestamp=0.08)
        self.video.add_frame(frame3)
        self.assertEqual(len(self.video.frames), 3)
        removed = self.video.remove_frame(3)
        self.assertEqual(removed.frame_number, 3)
        self.assertEqual(len(self.video.frames), 2)

    def test_tags_and_annotations(self):
        self.video.add_tag("test")
        self.assertTrue(self.video.has_tag("test"))
        self.assertTrue(self.video.remove_tag("test"))
        self.assertFalse(self.video.has_tag("test"))
        self.video.annotations["note"] = "important"
        self.assertEqual(self.video.annotations["note"], "important")

    def test_processing_methods(self):
        self.video.start_processing()
        self.assertIsNotNone(self.video.processing_start_time)
        self.video.finish_processing()
        self.assertIsNotNone(self.video.processing_end_time)
        self.video.add_processing_error("err")
        self.assertIn("err", self.video.processing_errors)
        summary = self.video.get_processing_summary()
        self.assertIn("total_frames", summary)

    def test_clone(self):
        clone = self.video.clone()
        self.assertEqual(clone.video_id, "vid1_copy")
        self.assertEqual(clone.metadata.file_path, self.video.metadata.file_path)
        self.assertEqual(len(clone.frames), len(self.video.frames))
        self.assertEqual(clone.global_field.length, self.video.global_field.length)


if __name__ == "__main__":
    unittest.main()
