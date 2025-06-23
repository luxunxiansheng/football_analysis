"""
Test utilities to reduce code duplication across test files.
"""

import cv2
import os
import numpy as np
from typing import List, Tuple

from football_ai.domain.data_models import VideoData, FrameData
from football_ai.detection.object_detection_processor import ObjectDetectionProcessor
from football_ai.tracking.track_processor import TrackProcessor


def get_test_video_path() -> str:
    """Get the standard test video path."""
    return os.path.abspath("input_videos/08fd33_4.mp4")


def get_test_model_path() -> str:
    """Get the standard test model path."""
    return os.path.abspath("models/detect/best.pt")


def create_test_processors() -> Tuple[ObjectDetectionProcessor, TrackProcessor]:
    """
    Create standard test processors with optimized settings.

    Returns:
        Tuple of (detection_processor, track_processor)
    """
    detection_processor = ObjectDetectionProcessor(get_test_model_path())
    track_processor = TrackProcessor(
        track_activation_threshold=0.15,
        lost_track_buffer=120,
        minimum_matching_threshold=0.95,
        frame_rate=30,
        minimum_consecutive_frames=1,
    )
    return detection_processor, track_processor


def load_test_frames(video_path: str, max_frames: int = 6) -> List[FrameData]:
    """
    Load a limited number of frames for testing.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load

    Returns:
        List of FrameData objects
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while frame_count < max_frames:
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
    return frames


def create_test_video_data(video_path: str, max_frames: int = 6) -> VideoData:
    """
    Create VideoData object for testing.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load

    Returns:
        VideoData object with test frames
    """
    frames = load_test_frames(video_path, max_frames)

    if not frames:
        raise ValueError(f"No frames loaded from {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame = frames[0].raw_frame
    height, width = frame.shape[:2]
    cap.release()

    return VideoData(
        video_path=video_path,
        frame_rate=fps,
        resolution=(width, height),
        duration=len(frames) / fps,
        frames=frames,
    )


def run_standard_pipeline_test(max_frames: int = 3) -> VideoData:
    """
    Run a standard pipeline test with detection and tracking.

    Args:
        max_frames: Maximum number of frames to process

    Returns:
        Processed VideoData
    """
    video_path = get_test_video_path()
    detection_processor, track_processor = create_test_processors()

    video_data = create_test_video_data(video_path, max_frames)
    detected = detection_processor.process(video_data)
    tracked = track_processor.process(detected)

    return tracked
