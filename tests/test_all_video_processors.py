import unittest
import os
from football_ai.core_models.video import Video
from football_ai.core_models.player import Player
from football_ai.sources.video.video_loader import VideoLoader
from football_ai.sources.video.processors.motion_analysis.speed_calculator import (
    ObjectMotionProcessor,
)
from football_ai.sources.video.processors.motion_analysis.camera_stabilizer import (
    CameraMotionProcessor,
)
from football_ai.sources.video.processors.coordinate_transformation.coordinate_transformer import (
    FieldTransformationProcessor,
)
from football_ai.sources.video.processors.team_classification.siglip_team_classifier import (
    SigLIPTeamAssignmentProcessor,
)

VIDEO_PATH = "input_videos/08fd33_4.mp4"
SIGLIP_MODEL_PATH = "models/embed/siglip-base-patch16-224"


class TestObjectMotionProcessor(unittest.TestCase):
    def test_object_motion(self):
        if not os.path.exists(VIDEO_PATH):
            self.skipTest(f"Test video not found: {VIDEO_PATH}")
        loader = VideoLoader(log_level="ERROR")
        video = loader.load_video(VIDEO_PATH, max_frames=2)
        # Add players and ball to each frame for testing
        for frame in video.frames:
            frame.players = {
                1: Player(track_id=1, team_id=1, pixel_position=(10, 10)),
                2: Player(track_id=2, team_id=2, pixel_position=(100, 100)),
            }
            from football_ai.core_models.ball import Ball

            frame.ball = Ball(track_id=1, pixel_position=(20, 20))
        processor = ObjectMotionProcessor()
        result = processor.process(video)
        for frame in result.frames:
            self.assertEqual(frame.players[1].object_position, (10, 10))
            self.assertEqual(frame.players[2].object_position, (100, 100))
            self.assertEqual(frame.ball.object_position, (20, 20))


class TestCameraMotionProcessor(unittest.TestCase):
    def test_camera_motion(self):
        if not os.path.exists(VIDEO_PATH):
            self.skipTest(f"Test video not found: {VIDEO_PATH}")
        loader = VideoLoader(log_level="ERROR")
        video = loader.load_video(VIDEO_PATH, max_frames=2)
        processor = CameraMotionProcessor()
        result = processor.process(video)
        self.assertIsInstance(result, Video)
        # We can't guarantee motion, but the processor should run without error


class TestFieldTransformationProcessor(unittest.TestCase):
    def test_field_transformation(self):
        if not os.path.exists(VIDEO_PATH):
            self.skipTest(f"Test video not found: {VIDEO_PATH}")
        loader = VideoLoader(log_level="ERROR")
        video = loader.load_video(VIDEO_PATH, max_frames=2)
        # Add players and ball to each frame for testing
        for frame in video.frames:
            frame.players = {
                1: Player(track_id=1, team_id=1, pixel_position=(10, 10)),
                2: Player(track_id=2, team_id=2, pixel_position=(100, 100)),
            }
            from football_ai.core_models.ball import Ball

            frame.ball = Ball(track_id=1, pixel_position=(20, 20))
        processor = FieldTransformationProcessor()
        result = processor.process(video)
        for frame in result.frames:
            self.assertIsNotNone(frame.players[1].field_position)
            self.assertIsNotNone(frame.players[2].field_position)
            self.assertIsNotNone(frame.ball.field_position)


class TestSigLIPTeamAssignmentProcessor(unittest.TestCase):
    def test_team_assignment(self):
        if not os.path.exists(VIDEO_PATH):
            self.skipTest(f"Test video not found: {VIDEO_PATH}")
        loader = VideoLoader(log_level="ERROR")
        video = loader.load_video(VIDEO_PATH, max_frames=1)
        # Add players to each frame for testing
        for frame in video.frames:
            frame.players = {
                1: Player(track_id=1, team_id=None, pixel_position=(10, 10)),
                2: Player(track_id=2, team_id=None, pixel_position=(100, 100)),
            }
        processor = SigLIPTeamAssignmentProcessor(
            device="cpu", n_clusters=2, model_path=SIGLIP_MODEL_PATH
        )
        result = processor.process(video)
        self.assertIsInstance(result, Video)
        # We can't guarantee assignment, but the processor should run without error


if __name__ == "__main__":
    unittest.main()
