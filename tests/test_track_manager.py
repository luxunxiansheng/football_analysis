import os
import sys
import tempfile
import pickle
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import supervision as sv

# Add src and config directories to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
)

from track_manager import TrackManager
from track_manager_config import TrackManagerConfig as config


class TestTrackManager(unittest.TestCase):
    def setUp(self):
        self.track_manager = TrackManager(config.model_path)

    def test_initialization(self):
        self.assertIsNotNone(self.track_manager)

        def test_detect_frames(self):
            """Test the _detect_frames method"""
            # Create mock frames
            mock_frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(5)]

            # Test with default parameters
            detections = self.track_manager._detect_frames(mock_frames)
            self.assertEqual(len(detections), len(mock_frames))

        def test_init_tracks_stub_reading(self):
            """Test reading tracks from stub file"""

            # Create a temporary stub file
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                stub_path = temp.name
                mock_tracks = {
                    "players": [{"1": {"bbox": [10, 20, 30, 40]}}],
                    "referees": [{}],
                    "ball": [{"1": {"bbox": [50, 60, 70, 80]}}],
                }
                pickle.dump(mock_tracks, temp)

            # Create minimal test data
            mock_frames = [np.zeros((720, 1280, 3), dtype=np.uint8)]
            mock_tracker = sv.ByteTrack()

            # Test reading from stub
            self.track_manager.init_tracks(
                tracker=mock_tracker,
                frames=mock_frames,
                read_from_stub=True,
                stub_path=stub_path,
            )

            # Check if tracks were loaded from stub
            self.assertEqual(self.track_manager.tracks, mock_tracks)

            # Clean up
            os.remove(stub_path)

        def test_init_tracks_detection_processing(self):
            """Test track initialization with detection processing"""

            # Create minimal test data
            mock_frames = [np.zeros((720, 1280, 3), dtype=np.uint8)]
            mock_tracker = sv.ByteTrack()

            # Create a mock detection
            mock_detection = MagicMock()
            mock_detection.names = {
                0: "player",
                1: "referee",
                2: "ball",
                3: "goalkeeper",
            }

            # Mock detection_supervision
            mock_det_supervision = MagicMock(spec=sv.Detections)
            mock_det_supervision.class_id = np.array(
                [0, 3, 1, 2]
            )  # player, goalkeeper, referee, ball

            # Mock detection with tracks
            mock_det_with_tracks = [
                (np.array([10, 20, 30, 40]), None, None, 0, 1),  # player
                (np.array([50, 60, 70, 80]), None, None, 1, 2),  # referee
            ]

            # Patch the necessary methods
            with patch.object(
                self.track_manager, "_detect_frames", return_value=[mock_detection]
            ), patch.object(
                sv.Detections, "from_ultralytics", return_value=mock_det_supervision
            ), patch.object(
                mock_tracker,
                "update_with_detections",
                return_value=mock_det_with_tracks,
            ):

                # Run the method
                self.track_manager.init_tracks(tracker=mock_tracker, frames=mock_frames)

                # Verify tracking data structure
                self.assertEqual(len(self.track_manager.tracks["players"]), 1)
                self.assertEqual(len(self.track_manager.tracks["referees"]), 1)
                self.assertEqual(len(self.track_manager.tracks["ball"]), 1)

        def test_init_tracks_stub_writing(self):
            """Test writing tracks to a stub file"""

            # Create temporary stub file
            stub_path = tempfile.mktemp()

            # Create minimal test data
            mock_frames = [np.zeros((720, 1280, 3), dtype=np.uint8)]
            mock_tracker = sv.ByteTrack()

            # Mock the detection process
            with patch.object(
                self.track_manager, "_detect_frames", return_value=[]
            ), patch.object(sv.Detections, "from_ultralytics"), patch.object(
                mock_tracker, "update_with_detections"
            ):

                # Run the method with stub writing
                self.track_manager.init_tracks(
                    tracker=mock_tracker, frames=mock_frames, stub_path=stub_path
                )

                # Verify stub file was created and contains track data
                self.assertTrue(os.path.exists(stub_path))
                with open(stub_path, "rb") as f:
                    saved_tracks = pickle.load(f)
                    self.assertEqual(saved_tracks, self.track_manager.tracks)

                # Clean up
                os.remove(stub_path)


if __name__ == "__main__":
    unittest.main()
