"""
Video Analysis Processors

This package contains all processors specific to video analysis.
These processors handle detection, tracking, assignment, and analysis
of football video content.
"""

from .detection import ObjectDetectionProcessor
from .tracking import TrackProcessor
from .motion import ObjectMotionProcessor, CameraMotionProcessor
from .transformation import FieldTransformationProcessor
from .assignment import SigLIPTeamAssignmentProcessor, BallAssignmentProcessor
from .analysis import SpeedProcessor, BallControlProcessor
from .rendering import RendererProcessor
from .storing import VideoWriterProcessor

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
