import unittest
import os
from football_ai.core_models.video import Video
from football_ai.core_models.player import Player
from football_ai.sources.video.video_loader import VideoLoader
from football_ai.sources.video.processors.team_classification.siglip_team_classifier import (
    SigLIPTeamAssignmentProcessor,
)

VIDEO_PATH = "input_videos/08fd33_4.mp4"
SIGLIP_MODEL_PATH = "models/embed/siglip-base-patch16-224"


class TestSigLIPTeamAssignmentProcessor(unittest.TestCase):
    def test_team_assignment(self):
        if not os.path.exists(VIDEO_PATH):
            self.skipTest(f"Test video not found: {VIDEO_PATH}")
        loader = VideoLoader(log_level="ERROR")
        video = loader.load_video(VIDEO_PATH, max_frames=1)
        # Add 10 players to the frame for robust clustering and UMAP
        for frame in video.frames:
            frame.players = {
                i: Player(track_id=i, team_id=None, pixel_position=(i * 10, i * 10))
                for i in range(1, 11)
            }
        processor = SigLIPTeamAssignmentProcessor(
            device="cpu", n_clusters=2, model_path=SIGLIP_MODEL_PATH
        )
        result = processor.process(video)
        self.assertIsInstance(result, Video)


if __name__ == "__main__":
    unittest.main()
