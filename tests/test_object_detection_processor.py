import cv2
import os
from football_ai.detection.object_detection_processor import ObjectDetectionProcessor
from football_ai.domain.data_models import VideoData, FrameData


def test_object_detection_processor_on_real_video():
    video_path = os.path.abspath("input_videos/08fd33_4.mp4")
    model_path = os.path.abspath("models/detect/best.pt")
    processor = ObjectDetectionProcessor(model_path)

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
        num_detections = (
            len(frame_data.detections) if frame_data.detections is not None else 0
        )
        print(f"Frame {frame_data.frame_number}: {num_detections} detections")
        assert isinstance(frame_data.detections, list)
    assert any(
        (f.detections is not None and len(f.detections) > 0) for f in result.frames
    )


def run_tests():
    test_object_detection_processor_on_real_video()
    print("ObjectDetectionProcessor real video test passed.")


if __name__ == "__main__":
    run_tests()
