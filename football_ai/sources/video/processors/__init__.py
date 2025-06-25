"""
Video Analysis Processors

This package contains all processors specific to video analysis.
These processors handle detection, tracking, assignment, and analysis
of football video content.
"""

from .object_detection import ObjectDetectionProcessor
from .object_tracking import TrackProcessor
from .motion_analysis import ObjectMotionProcessor, CameraMotionProcessor
from .coordinate_transformation import FieldTransformationProcessor
from .team_classification import SigLIPTeamAssignmentProcessor, BallAssignmentProcessor
from .match_analysis import SpeedProcessor, BallControlProcessor
from .video_rendering import RendererProcessor
from .video_export import VideoWriterProcessor

__all__ = [
    "ObjectDetectionProcessor",
    "TrackProcessor",
    "ObjectMotionProcessor",
    "CameraMotionProcessor",
    "FieldTransformationProcessor",
    "SigLIPTeamAssignmentProcessor",
    "BallAssignmentProcessor",
    "SpeedProcessor",
    "BallControlProcessor",
    "RendererProcessor",
    "VideoWriterProcessor",
]
