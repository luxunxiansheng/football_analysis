"""
Video processing utilities.
"""

import cv2
import os
from typing import Tuple, Optional
from pathlib import Path


def validate_video_path(video_path: str) -> bool:
    """
    Validate if a video file exists and is readable.

    Args:
        video_path: Path to video file

    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(video_path):
        return False

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        cap.release()
        return True
    except Exception:
        return False


def get_video_info(video_path: str) -> Tuple[int, int, float, int]:
    """
    Get basic video information.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height, fps, frame_count)

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return width, height, fps, frame_count
    finally:
        cap.release()


def create_video_writer(
    output_path: str, width: int, height: int, fps: float, fourcc: str = "mp4v"
) -> cv2.VideoWriter:
    """
    Create a video writer with standard settings.

    Args:
        output_path: Output video path
        width: Video width
        height: Video height
        fps: Frames per second
        fourcc: Video codec fourcc code

    Returns:
        Configured VideoWriter
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc_code = cv2.VideoWriter.fourcc(*fourcc)
    writer = cv2.VideoWriter(output_path, fourcc_code, fps, (width, height))

    if not writer.isOpened():
        raise ValueError(f"Failed to create video writer for: {output_path}")

    return writer
