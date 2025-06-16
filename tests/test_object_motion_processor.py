import cv2
import os
from football_ai.detection.object_detection_processor import ObjectDetectionProcessor
from football_ai.tracking.track_processor import TrackProcessor
from football_ai.motion.object_motion_processor import ObjectMotionProcessor
from football_ai.domain.data_models import VideoData, FrameData


def test_object_motion_processor():
    video_path = os.path.abspath("input_videos/08fd33_4.mp4")
    model_path = os.path.abspath("models/detect/best.pt")
    detection_processor = ObjectDetectionProcessor(model_path)
    track_processor = TrackProcessor()
    motion_processor = ObjectMotionProcessor()

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
    result = motion_processor.process(tracked)

    for frame_data in result.frames:
        positions = (
            frame_data.metadata.get("object_positions", {})
            if frame_data.metadata
            else {}
        )
        print(f"Frame {frame_data.frame_number}: object_positions={positions}")
        assert isinstance(positions, dict)
        for obj_id, pos in positions.items():
            assert isinstance(pos, list) and len(pos) == 2
    print("ObjectMotionProcessor test passed.")


if __name__ == "__main__":
    test_object_motion_processor()
