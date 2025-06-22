import cv2
import os
import numpy as np
from football_ai.detection.object_detection_processor import ObjectDetectionProcessor
from football_ai.tracking.track_processor import TrackProcessor
from football_ai.motion.object_motion_processor import ObjectMotionProcessor
from football_ai.assignment.team_assignment_processor import SigLIPTeamAssignmentProcessor as TeamAssignmentProcessor

from football_ai.domain.data_models import VideoData, FrameData

def test_team_assignment_processor():
    video_path = os.path.abspath("input_videos/08fd33_4.mp4")
    detect_model_path = os.path.abspath("models/detect/best.pt")
    detection_processor = ObjectDetectionProcessor(detect_model_path)

    team_assignment_model_path = os.path.abspath("models/embed/team_assignment_model")
    if not os.path.exists(team_assignment_model_path):
        os.makedirs(team_assignment_model_path)

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while frame_count < 6:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(
            FrameData(
                frame_number=frame_count,
                timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                raw_frame=frame,
            )
        )
        frame_count += 1
    cap.release()

    video_data = VideoData(
        video_path=video_path,
        frame_rate=cap.get(cv2.CAP_PROP_FPS),
        resolution=(frame.shape[1], frame.shape[0]) if frames else (0, 0),
        duration=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
        frames=frames,
    )

    # Run the detection processor
    video_data = detection_processor.process(video_data)

    # train the team assignment processor with first 3 frames
    team_assignment_processor = TeamAssignmentProcessor(
        model_path=team_assignment_model_path, device="cuda"
    )

    print(f"Training team assignment processor...")
    team_assignment_processor.train(video_data, batch_size=2)
    print(f"Training completed!")

    print(f"Processing video data for team assignments...")
    processed_video_data = team_assignment_processor.process(video_data)

    # Validate results
    total_players = 0
    team_assignments = {}

    for frame_idx, frame in enumerate(processed_video_data.frames):
        if frame.detections:
            frame_players = [
                d
                for d in frame.detections
                if d.metadata
                and "team_id" in d.metadata
                and d.metadata["team_id"] is not None
            ]
            total_players += len(frame_players)

            for detection in frame_players:
                team_id = detection.metadata["team_id"]
                if team_id not in team_assignments:
                    team_assignments[team_id] = 0
                team_assignments[team_id] += 1

            print(f"Frame {frame_idx}: {len(frame_players)} players assigned to teams")

    print(f"\nTest Results:")
    print(f"Total players processed: {total_players}")
    print(f"Team assignments: {team_assignments}")

    if team_assignments:
        print("✅ Team assignment processor test PASSED!")
    else:
        print("❌ Team assignment processor test FAILED - No team assignments found!")

    return processed_video_data


if __name__ == "__main__":
    test_team_assignment_processor()
