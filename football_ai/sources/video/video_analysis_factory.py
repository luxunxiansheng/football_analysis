"""
Video Analysis Factory Functions - Convenient factory functions for creating video analysis components.

This module provides factory functions that make it easy to create video analysis
components with sensible defaults, replacing the heavy config-based approach.
"""

from typing import Optional
from .video_analysis_processor import VideoAnalysisProcessor


def create_demo_processor(
    model_path: str, input_video_path: str, output_video_path: str
) -> VideoAnalysisProcessor:
    """
    Create a VideoAnalysisProcessor for demo/testing purposes.

    Args:
        model_path: Path to YOLO detection model
        input_video_path: Input video file path
        output_video_path: Output video file path

    Returns:
        VideoAnalysisProcessor configured for demo usage
    """
    return VideoAnalysisProcessor(
        model_path=model_path,
        confidence_threshold=0.3,
        device="cuda",
        log_level="INFO",
        enable_team_classification=True,
        enable_ball_tracking=True,
    )


def create_high_accuracy_processor(
    model_path: str, team_model_path: Optional[str] = None
) -> VideoAnalysisProcessor:
    """
    Create a VideoAnalysisProcessor optimized for high accuracy.

    Args:
        model_path: Path to YOLO detection model
        team_model_path: Optional path to team classification model

    Returns:
        VideoAnalysisProcessor configured for high accuracy
    """
    return VideoAnalysisProcessor(
        model_path=model_path,
        confidence_threshold=0.1,  # Very sensitive
        iou_threshold=0.3,  # Less overlap tolerance
        device="cuda",
        track_threshold=0.2,  # Lower tracking threshold
        track_buffer=120,  # Longer buffer for occlusions
        max_detections=2000,  # More detections
        log_level="INFO",
        enable_team_classification=True,
        enable_ball_tracking=True,
        team_model_path=team_model_path,
    )


def create_fast_processor(model_path: str) -> VideoAnalysisProcessor:
    """
    Create a VideoAnalysisProcessor optimized for speed.

    Args:
        model_path: Path to YOLO detection model

    Returns:
        VideoAnalysisProcessor configured for fast processing
    """
    return VideoAnalysisProcessor(
        model_path=model_path,
        confidence_threshold=0.5,  # Less sensitive
        iou_threshold=0.6,  # More overlap tolerance
        device="cuda",
        track_threshold=0.6,  # Higher tracking threshold
        track_buffer=30,  # Shorter buffer
        max_detections=500,  # Fewer detections
        log_level="WARNING",  # Less logging
        enable_team_classification=False,  # Disable heavy features
        enable_ball_tracking=True,  # Keep ball tracking
    )


def create_broadcast_processor(
    model_path: str, team_model_path: Optional[str] = None
) -> VideoAnalysisProcessor:
    """
    Create a VideoAnalysisProcessor optimized for broadcast analysis.

    Args:
        model_path: Path to YOLO detection model
        team_model_path: Optional path to team classification model

    Returns:
        VideoAnalysisProcessor configured for broadcast quality
    """
    return VideoAnalysisProcessor(
        model_path=model_path,
        confidence_threshold=0.25,  # Balanced sensitivity
        iou_threshold=0.45,  # Standard overlap
        device="cuda",
        track_threshold=0.35,  # Balanced tracking
        track_buffer=90,  # Medium buffer
        max_detections=1500,  # Broadcast quality
        log_level="INFO",
        enable_team_classification=True,
        enable_ball_tracking=True,
        team_model_path=team_model_path,
    )
