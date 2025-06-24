import cv2
import os
from football_ai.sources.video.processors.motion.camera_motion_processor import CameraMotionProcessor
from football_ai.domain.data_models import VideoData, FrameData


def test_camera_motion_processor():
    video_path = os.path.abspath("input_videos/08fd33_4.mp4")
    processor = CameraMotionProcessor()

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

    result = processor.process(video_data)

    for frame_data in result.frames:
        metadata = frame_data.metadata if frame_data.metadata is not None else {}
        movement = metadata.get("camera_movement", None)
        print(f"Frame {frame_data.frame_number}: camera_movement={movement}")
        assert movement is not None
        # Flatten if movement is a list of lists with shape (1, 2)
        if (
            isinstance(movement, list)
            and len(movement) == 1
            and isinstance(movement[0], list)
            and len(movement[0]) == 2
        ):
            movement = movement[0]
        assert isinstance(movement, list) and len(movement) == 2
    print("CameraMotionProcessor test passed.")


if __name__ == "__main__":
    test_camera_motion_processor()
