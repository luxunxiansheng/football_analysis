import unittest
import os
from football_ai.core_models.video import Video
from football_ai.sources.video.video_loader import VideoLoader
from football_ai.sources.video.processors.object_detection.yolo_detector import (
    ObjectDetectionProcessor,
)
from football_ai.sources.video.processors.object_tracking.byte_tracker import (
    TrackProcessor,
)
from football_ai.core_models.player import Player

VIDEO_PATH = "input_videos/08fd33_4.mp4"
MODEL_PATH = "models/detect/yolo11n.pt"


class TestObjectDetectionProcessor(unittest.TestCase):
    def test_object_detection(self):
        if not os.path.exists(VIDEO_PATH):
            self.skipTest(f"Test video not found: {VIDEO_PATH}")
        loader = VideoLoader(log_level="ERROR")
        video = loader.load_video(VIDEO_PATH, max_frames=2)
        processor = ObjectDetectionProcessor(
            model_path="models/detect/best.pt",  # Use the project's best YOLO model
            confidence_threshold=0.05,  # Lower threshold for more detections
            device="cpu",
        )
        result = processor.process(video)
        self.assertIsInstance(result, Video)
        self.assertGreaterEqual(len(result.frames), 1)
        # Check that at least one frame has players or ball detected
        found = False
        # Print detection results for debugging
        print("Detection results per frame:")
        for idx, frame in enumerate(result.frames):
            print(
                f"Frame {idx}: players={list(frame.players.keys())}, ball={'yes' if frame.ball else 'no'}"
            )
            if frame.players:
                found = True
            if frame.ball is not None:
                found = True
        self.assertTrue(
            found,
            "No players or ball detected in any frame. See printed results above.",
        )


class TestTrackProcessor(unittest.TestCase):
    def test_track_processor(self):
        if not os.path.exists(VIDEO_PATH):
            self.skipTest(f"Test video not found: {VIDEO_PATH}")
        loader = VideoLoader(log_level="ERROR")
        video = loader.load_video(VIDEO_PATH, max_frames=2)
        # Simulate detection by adding two players to each frame for real-data-like clustering
        for frame in video.frames:
            frame.players = {
                1: Player(track_id=1, team_id=1, pixel_position=(10, 10)),
                2: Player(track_id=2, team_id=2, pixel_position=(100, 100)),
            }
            frame.goalkeepers = {}
            frame.referees = {}
        processor = TrackProcessor()
        result = processor.process(video)
        self.assertIsInstance(result, Video)
        self.assertGreaterEqual(len(result.frames), 1)


class TestEndToEndVideoPipeline(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        if not os.path.exists(VIDEO_PATH):
            self.skipTest(f"Test video not found: {VIDEO_PATH}")
        loader = VideoLoader(log_level="ERROR")
        # Increase max_frames to 10 for more real player crops
        video = loader.load_video(VIDEO_PATH, max_frames=10)
        # Step 1: Object Detection
        detection_processor = ObjectDetectionProcessor(
            model_path="models/detect/best.pt",
            confidence_threshold=0.05,
            device="cuda",
        )
        detected_video = detection_processor.process(video)
        print("[End-to-End] Detection results:")
        for idx, frame in enumerate(detected_video.frames):
            print(
                f"Frame {idx}: players={list(frame.players.keys())}, ball={'yes' if frame.ball else 'no'}"
            )
        self.assertTrue(
            any(frame.players for frame in detected_video.frames),
            "No players detected in any frame after detection.",
        )
        # Step 2: Tracking
        track_processor = TrackProcessor()
        tracked_video = track_processor.process(detected_video)
        print("[End-to-End] Tracking results:")
        for idx, frame in enumerate(tracked_video.frames):
            print(f"Frame {idx}: players={list(frame.players.keys())}")
        self.assertTrue(
            any(frame.players for frame in tracked_video.frames),
            "No players present in any frame after tracking.",
        )
        self.assertIsInstance(tracked_video, Video)
        self.assertGreaterEqual(len(tracked_video.frames), 1)
        # Step 3: Add more processors (e.g., Ball Control, Camera Motion, Field Transformation)
        try:
            from football_ai.sources.video.processors.motion_analysis.camera_stabilizer import (
                CameraMotionProcessor,
            )
            from football_ai.sources.video.processors.motion_analysis.speed_calculator import (
                ObjectMotionProcessor,
            )
            from football_ai.sources.video.processors.coordinate_transformation.coordinate_transformer import (
                FieldTransformationProcessor,
            )
        except ImportError:
            self.skipTest("One or more additional processors could not be imported.")
        # Camera motion
        camera_motion_processor = CameraMotionProcessor()
        camera_motion_video = camera_motion_processor.process(tracked_video)
        self.assertIsInstance(camera_motion_video, Video)
        # Object motion (speed)
        object_motion_processor = ObjectMotionProcessor()
        object_motion_video = object_motion_processor.process(camera_motion_video)
        self.assertIsInstance(object_motion_video, Video)
        # Field transformation
        field_transformation_processor = FieldTransformationProcessor()
        field_transformed_video = field_transformation_processor.process(
            object_motion_video
        )
        self.assertIsInstance(field_transformed_video, Video)
        # Step 4: Ball Control Processor
        # Removed BallControlProcessor import (deprecated, use Ball class instead)
        # Step 5: Team Assignment (SigLIP)
        try:
            from football_ai.sources.video.processors.team_classification.siglip_team_classifier import (
                SigLIPTeamAssignmentProcessor,
            )
        except ImportError:
            self.skipTest("SigLIPTeamAssignmentProcessor could not be imported.")
        siglip_processor = SigLIPTeamAssignmentProcessor(
            model_path="models/embed/siglip-base-patch16-224/",
            device="cpu",
        )
        team_assigned_video = siglip_processor.process(ball_control_video)
        self.assertIsInstance(team_assigned_video, Video)
        print(
            "[End-to-End] All processors including Ball Control and Team Assignment completed."
        )
        for idx, frame in enumerate(team_assigned_video.frames):
            self.assertTrue(hasattr(frame, "players"))
            self.assertTrue(hasattr(frame, "ball"))

    def test_end_to_end_detection_and_tracking(self):
        if not os.path.exists(VIDEO_PATH):
            self.skipTest(f"Test video not found: {VIDEO_PATH}")
        loader = VideoLoader(log_level="ERROR")
        video = loader.load_video(VIDEO_PATH, max_frames=2)
        # Step 1: Object Detection
        detection_processor = ObjectDetectionProcessor(
            model_path="models/detect/best.pt",
            confidence_threshold=0.05,
            device="cpu",
        )
        detected_video = detection_processor.process(video)
        self.assertIsInstance(detected_video, Video)
        self.assertGreaterEqual(len(detected_video.frames), 1)
        found = False
        for frame in detected_video.frames:
            if frame.players or frame.ball is not None:
                found = True
        self.assertTrue(found, "No players or ball detected in any frame.")
        # Step 2: Tracking
        track_processor = TrackProcessor()
        tracked_video = track_processor.process(detected_video)
        self.assertIsInstance(tracked_video, Video)
        self.assertGreaterEqual(len(tracked_video.frames), 1)
        # Check that tracked players have track_ids
        tracked_found = False
        for frame in tracked_video.frames:
            for player in frame.players.values():
                if hasattr(player, "track_id") and player.track_id is not None:
                    tracked_found = True
        self.assertTrue(tracked_found, "No tracked players with track_id found.")


if __name__ == "__main__":
    unittest.main()
