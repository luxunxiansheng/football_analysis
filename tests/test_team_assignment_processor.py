import cv2
import os
import numpy as np
from football_ai.detection.object_detection_processor import ObjectDetectionProcessor
from football_ai.tracking.track_processor import TrackProcessor
from football_ai.motion.object_motion_processor import ObjectMotionProcessor
from football_ai.assignment.team_assignment_processor import TeamAssignmentProcessor
from football_ai.domain.data_models import VideoData, FrameData


def test_team_assignment_processor():
    video_path = os.path.abspath("input_videos/08fd33_4.mp4")
    model_path = os.path.abspath("models/detect/best.pt")
    detection_processor = ObjectDetectionProcessor(model_path)
    track_processor = TrackProcessor()
    motion_processor = ObjectMotionProcessor()
    team_processor = TeamAssignmentProcessor()

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while frame_count < 3:
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

    detected = detection_processor.process(video_data)
    tracked = track_processor.process(detected)
    motioned = motion_processor.process(tracked)
    result = team_processor.process(motioned)

    for frame_data in result.frames:
        teams = [getattr(d, "team", None) for d in (frame_data.detections or [])]
        print(f"Frame {frame_data.frame_number}: team assignments={teams}")
        # At least some players/goalkeepers should have a team assigned (1 or 2)
        assert any(t in (1, 2) for t in teams if t is not None)
    print("TeamAssignmentProcessor test passed.")


if __name__ == "__main__":
    test_team_assignment_processor()
