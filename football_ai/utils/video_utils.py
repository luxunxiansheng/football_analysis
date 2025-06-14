"""
Video Utilities Module

This module provides utility functions for video processing including
reading, writing, and frame manipulation.
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional, List
import os


def read_video_frames(video_path: str) -> Generator[np.ndarray, None, None]:
    """
    Generator that yields video frames one by one.

    Args:
        video_path: Path to the input video file

    Yields:
        Video frames as numpy arrays
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def get_video_properties(video_path: str) -> dict:
    """
    Get video properties like FPS, frame count, resolution.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary with video properties
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        properties = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        }
        return properties
    finally:
        cap.release()


class VideoWriter:
    """
    Modern video writer with context manager support.
    """

    def __init__(
        self,
        output_path: str,
        fps: float = 24.0,
        frame_size: Optional[Tuple[int, int]] = None,
        fourcc: str = "mp4v",
    ):
        """
        Initialize video writer.

        Args:
            output_path: Path for output video
            fps: Frames per second
            frame_size: (width, height) of output video
            fourcc: Video codec fourcc code
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = cv2.VideoWriter.fourcc(*fourcc)
        self.writer: Optional[cv2.VideoWriter] = None
        self.is_initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def write_frame(self, frame: np.ndarray):
        """Write a single frame to the video."""
        if not self.is_initialized:
            # Initialize writer with first frame dimensions
            h, w = frame.shape[:2]
            if self.frame_size is None:
                self.frame_size = (w, h)

            self.writer = cv2.VideoWriter(
                self.output_path, self.fourcc, self.fps, self.frame_size
            )
            self.is_initialized = True

        if self.writer is not None:
            # Resize frame if necessary
            if frame.shape[:2][::-1] != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)

            self.writer.write(frame)

    def release(self):
        """Release the video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.is_initialized = False


def resize_frame(
    frame: np.ndarray, target_size: Tuple[int, int], maintain_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize frame to target size.

    Args:
        frame: Input frame
        target_size: (width, height) target size
        maintain_aspect_ratio: Whether to maintain aspect ratio

    Returns:
        Resized frame
    """
    if not maintain_aspect_ratio:
        return cv2.resize(frame, target_size)

    h, w = frame.shape[:2]
    target_w, target_h = target_size

    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_w / w, target_h / h)

    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h))

    # Create canvas with target size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Center the resized frame on canvas
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 1,
    max_frames: Optional[int] = None,
) -> List[str]:
    """
    Extract frames from video and save as images.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        frame_interval: Extract every nth frame
        max_frames: Maximum number of frames to extract

    Returns:
        List of paths to extracted frame images
    """
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = []
    frame_count = 0
    extracted_count = 0

    for frame in read_video_frames(video_path):
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{extracted_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            extracted_count += 1

            if max_frames and extracted_count >= max_frames:
                break

        frame_count += 1

    return frame_paths


def create_video_from_frames(
    frame_paths: List[str], output_path: str, fps: float = 24.0
) -> bool:
    """
    Create video from a list of frame image paths.

    Args:
        frame_paths: List of paths to frame images
        output_path: Path for output video
        fps: Frames per second

    Returns:
        True if successful, False otherwise
    """
    if not frame_paths:
        return False

    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        return False

    h, w = first_frame.shape[:2]

    with VideoWriter(output_path, fps, (w, h)) as writer:
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                writer.write_frame(frame)

    return True


def crop_frame(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop frame using bounding box coordinates.

    Args:
        frame: Input frame
        bbox: Bounding box (x1, y1, x2, y2)

    Returns:
        Cropped frame
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # Ensure coordinates are within frame bounds
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(x1, min(x2, w))
    y2 = max(y1, min(y2, h))

    return frame[y1:y2, x1:x2]


def apply_blur(frame: np.ndarray, blur_strength: int = 15) -> np.ndarray:
    """
    Apply Gaussian blur to frame.

    Args:
        frame: Input frame
        blur_strength: Strength of blur (must be odd)

    Returns:
        Blurred frame
    """
    if blur_strength % 2 == 0:
        blur_strength += 1  # Ensure odd number

    return cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)


def adjust_brightness_contrast(
    frame: np.ndarray, brightness: int = 0, contrast: float = 1.0
) -> np.ndarray:
    """
    Adjust brightness and contrast of frame.

    Args:
        frame: Input frame
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast multiplier (0.0 to 3.0)

    Returns:
        Adjusted frame
    """
    adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    return adjusted
