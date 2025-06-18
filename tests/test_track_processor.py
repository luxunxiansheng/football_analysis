import cv2
import os
from football_ai.detection.object_detection_processor import ObjectDetectionProcessor
from football_ai.tracking.track_processor import TrackProcessor
from football_ai.domain.data_models import VideoData, FrameData


def test_track_processor_with_supervision():
    video_path = os.path.abspath("input_videos/08fd33_4.mp4")
    model_path = os.path.abspath("models/detect/best.pt")
    detection_processor = ObjectDetectionProcessor(model_path)
    track_processor = TrackProcessor(
        track_activation_threshold=0.15,
        lost_track_buffer=120,
        minimum_matching_threshold=0.95,
        frame_rate=30,
        minimum_consecutive_frames=1,
    )

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

    # Step 1: Detect objects
    detected = detection_processor.process(video_data)
    # Step 2: Track objects
    tracked = track_processor.process(detected)

    for frame_data in tracked.frames:
        ids = [d.track_id for d in (frame_data.detections or [])]
        print(f"Frame {frame_data.frame_number}: track_ids={ids}")
        assert all((tid is None or isinstance(tid, int)) for tid in ids)
    print("TrackProcessor with Supervision test passed.")


if __name__ == "__main__":
    test_track_processor_with_supervision()
