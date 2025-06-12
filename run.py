# Standard library imports
import numpy as np

# Local application imports
from camera_movement_estimator import CameraMovementEstimator
from player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from team_assigner import TeamAssigner
from trackers import Tracker
from utils import read_video, save_video
from view_transformer import ViewTransformer


def main():
    # Read Video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Initialize Tracker
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
    )
    # Get object positions
    tracker.add_position_to_tracks(tracks)


    # Save video
    save_video(output_video_frames, "output_videos/output_video.mp4")


if __name__ == "__main__":
    main()
