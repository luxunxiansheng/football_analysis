"""
Video Loader - Handles loading and preparation of video data for analysis.

This module is responsible for loading video files and converting them
into a format suitable for analysis processors.
"""

from pathlib import Path
from typing import List, Optional
import cv2

from ...utilities import create_progress_bar, setup_logger
from ...core_models.video import Video
from ...core_models.frame import Frame


class VideoLoader:
    """
    Video loader that creates Video objects from file paths.

    This class handles the low-level video loading and frame extraction,
    preparing video data for analysis by the video pipeline.
    """

    def __init__(self, log_level: str = "INFO"):
        self.logger = setup_logger("VideoLoader", log_level)

    def load_video(self, video_path: str, max_frames: Optional[int] = None) -> Video:
        """
        Load video from file and create Video object.

        Args:
            video_path: Path to input video file
            max_frames: Optional limit on number of frames to load

        Returns:
            Video object with loaded frames
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.logger.info(f"Loading video: {video_path}")

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / frame_rate if frame_rate > 0 else 0

        # Limit frames if specified
        frames_to_load = min(total_frames, max_frames) if max_frames else total_frames

        self.logger.info(
            f"Video properties: {width}x{height}, {frame_rate:.2f} FPS, "
            f"{total_frames} frames ({duration:.2f}s)"
        )

        # Load frames with progress bar
        frames = []
        progress_bar = create_progress_bar(
            total=frames_to_load, desc="Loading frames", unit="frames"
        )

        frame_number = 0
        while frame_number < frames_to_load:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frames.append(
                Frame(frame_number=frame_number, timestamp=timestamp, raw_frame=frame)
            )
            frame_number += 1
            progress_bar.update(1)

        progress_bar.close()
        cap.release()

        # Create Video object
        video = Video(
            video_id=Path(video_path).stem,
            video_path=video_path,
        )

        # Set metadata
        video.metadata.frame_rate = frame_rate
        video.metadata.resolution = (width, height)
        video.metadata.duration = duration
        video.frames = frames

        self.logger.info(f"Loaded {len(frames)} frames for processing")
        return video

    def validate_video_file(self, video_path: str) -> bool:
        """
        Validate that a video file can be opened and read.

        Args:
            video_path: Path to video file

        Returns:
            True if video is valid, False otherwise
        """
        try:
            if not Path(video_path).exists():
                return False

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            # Try to read first frame
            ret, _ = cap.read()
            cap.release()

            return ret
        except Exception:
            return False

    def get_video_info(self, video_path: str) -> dict:
        """
        Get basic information about a video file without loading frames.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video properties
        """
        if not self.validate_video_file(video_path):
            raise ValueError(f"Invalid video file: {video_path}")

        cap = cv2.VideoCapture(video_path)

        info = {
            "path": video_path,
            "frame_rate": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            "codec": cap.get(cv2.CAP_PROP_FOURCC),
        }

        cap.release()
        return info
