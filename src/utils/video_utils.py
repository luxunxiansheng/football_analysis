from typing import List
import cv2
import numpy as np


def read_video(video_path: str) -> List[np.ndarray]:
    """
    Read all frames from a video file.

    Args:
        video_path (str): Path to the input video file

    Returns:
        List[np.ndarray]: List of video frames as numpy arrays

    Raises:
        FileNotFoundError: If the video file cannot be opened
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def save_video(
    output_video_frames: List[np.ndarray],
    output_video_path: str,
    fps: int = 24,
    codec: str = "mp4v",
) -> None:
    """
    Save a list of frames as a video file.

    Args:
        output_video_frames (List[np.ndarray]): List of video frames to save
        output_video_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video (default: 24)
        codec (str): Video codec to use (default: 'mp4v')

    Raises:
        ValueError: If the frames list is empty
        RuntimeError: If the video writer cannot be initialized
    """
    if not output_video_frames:
        raise ValueError("No frames to save.")

    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in output_video_frames:
        out.write(frame)
    out.release()
