"""
Video Analysis Pipeline - Orchestrates all video processing components.

This module contains the video analysis pipeline that processes videos
using all the detection, tracking, and analysis processors.
"""

from typing import Optional, List

from ...core_models.video import Video
from ...core_models.interfaces import Processor
from ...utilities import setup_logger, create_progress_bar

# Import all video analysis processors
from .processors.object_detection.yolo_detector import ObjectDetectionProcessor
from .processors.object_tracking.byte_tracker import TrackProcessor
from .processors.motion_analysis.speed_calculator import ObjectMotionProcessor
from .processors.motion_analysis.camera_stabilizer import CameraMotionProcessor
from .processors.coordinate_transformation.coordinate_transformer import (
    FieldTransformationProcessor,
)
from .processors.team_classification.siglip_team_classifier import (
    SigLIPTeamAssignmentProcessor,
)
from .processors.team_classification.possession_analyzer import BallAssignmentProcessor
from .processors.domain_conversion.detection_converter import (
    DetectionConverterProcessor,
)
from .processors.match_analysis.speed_analyzer import SpeedProcessor
from .processors.match_analysis.possession_tracker import BallControlProcessor
from .processors.video_rendering.video_annotator import RendererProcessor
from .processors.video_export.video_exporter import VideoWriterProcessor


class VideoPipeline:
    """
    Video analysis pipeline that processes videos through all analysis stages.

    This class orchestrates the video processing workflow and provides
    video analysis capabilities within the game-centric architecture.
    """

    def __init__(
        self,
        # Core required parameters
        model_path: str,
        # Detection parameters
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        # Tracking parameters
        track_threshold: float = 0.4,
        track_buffer: int = 60,
        # Processing parameters
        max_detections: int = 1000,
        log_level: str = "INFO",
        # Feature toggles
        enable_team_classification: bool = True,
        enable_ball_tracking: bool = True,
        enable_motion_analysis: bool = True,
        enable_field_transformation: bool = True,
        enable_video_rendering: bool = True,
        # Model paths
        team_model_path: str = "models/embed/siglip-base-patch16-224",
        field_model_path: str = "models/pose/best.pt",
    ):
        """
        Initialize VideoPipeline with explicit parameters.

        Args:
            model_path: Path to main YOLO detection model
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            device: Device to run on ("cuda", "cpu", "mps")
            track_threshold: Tracking confidence threshold
            track_buffer: Frames to keep lost tracks
            max_detections: Max detections per frame
            log_level: Logging level
            enable_team_classification: Enable team color classification
            enable_ball_tracking: Enable ball tracking
            enable_motion_analysis: Enable motion analysis
            enable_field_transformation: Enable field coordinate transformation
            enable_video_rendering: Enable video annotation rendering
            team_model_path: Path to team classification model
            field_model_path: Path to field detection model
        """
        # Store parameters
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.track_threshold = track_threshold
        self.track_buffer = track_buffer
        self.max_detections = max_detections
        self.enable_team_classification = enable_team_classification
        self.enable_ball_tracking = enable_ball_tracking
        self.enable_motion_analysis = enable_motion_analysis
        self.enable_field_transformation = enable_field_transformation
        self.enable_video_rendering = enable_video_rendering
        self.team_model_path = team_model_path
        self.field_model_path = field_model_path

        # Setup logging
        self.logger = setup_logger("VideoPipeline", log_level)
        self.processors: List[Processor] = []
        self._build_processors()

    def _build_processors(self) -> None:
        """Build the video processing pipeline based on configuration."""
        self.processors = []

        # Core detection and tracking - using explicit parameters
        self.processors.append(
            ObjectDetectionProcessor(
                model_path=self.model_path,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
                device=self.device,
                max_detections=self.max_detections,
            )
        )

        self.processors.append(
            TrackProcessor(
                track_activation_threshold=self.track_threshold,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=0.75,  # Reasonable default
                frame_rate=30,  # Standard video frame rate
                minimum_consecutive_frames=1,
                min_track_length=5,
            )
        )

        # Motion analysis - if enabled
        if self.enable_motion_analysis:
            self.processors.append(ObjectMotionProcessor())
            self.processors.append(CameraMotionProcessor())

        # Field transformation - if enabled
        if self.enable_field_transformation:
            self.processors.append(FieldTransformationProcessor())

        # Team assignment - if enabled
        if self.enable_team_classification:
            team_assignment_processor = SigLIPTeamAssignmentProcessor(
                model_path=self.team_model_path,
                batch_size=32,  # Reasonable default
                device=self.device,
            )
            self.processors.append(team_assignment_processor)

        # Convert detections to domain objects (always needed for game analysis)
        self.processors.append(DetectionConverterProcessor())

        # Ball tracking - if enabled
        if self.enable_ball_tracking:
            self.processors.append(
                BallAssignmentProcessor(
                    max_distance=50.0  # Reasonable default distance
                )
            )
            self.processors.append(BallControlProcessor())

        # Analysis processors
        self.processors.append(SpeedProcessor())

        # Note: Video rendering processors (RendererProcessor, VideoWriterProcessor)
        # are added dynamically in process_video() when output_path is provided
        # because they require an output_path parameter

        self.logger.info(f"Built video pipeline with {len(self.processors)} processors")

    def process_video(self, video: Video, output_path: Optional[str] = None) -> Video:
        """
        Process a video through the complete analysis pipeline.

        Args:
            video: Video object to process
            output_path: Optional path for output video

        Returns:
            Processed Video object with analysis results
        """
        try:
            self.logger.info("Starting video analysis pipeline")

            # Create a copy of processors list and add output processors if needed
            processors_to_run = self.processors[:]

            # Add rendering and output processors if output_path is provided
            if output_path:
                try:
                    # Add renderer processor
                    renderer = RendererProcessor(output_path=output_path)
                    processors_to_run.append(renderer)

                    # Add video writer processor
                    writer = VideoWriterProcessor(output_path=output_path)
                    processors_to_run.append(writer)

                    self.logger.info(f"Added output processors for: {output_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to add output processors: {e}")

            # Process through pipeline
            progress_bar = create_progress_bar(
                iterable=processors_to_run,
                desc="Processing video pipeline",
                unit="processor",
                disable=False,  # Always show progress for now
            )

            for i, processor in enumerate(progress_bar):
                processor_name = processor.__class__.__name__
                progress_bar.set_description(f"Running {processor_name}")

                self.logger.info(
                    f"Running processor {i+1}/{len(processors_to_run)}: {processor_name}"
                )

                try:
                    video = processor.process(video)
                    self.logger.debug(f"✓ {processor_name} completed successfully")
                except Exception as e:
                    # For now, continue on errors but log them
                    self.logger.error(f"⚠ {processor_name} failed: {e}")
                    continue

            progress_bar.close()

            self.logger.info("✅ Video analysis pipeline completed successfully")
            return video

        except Exception as e:
            self.logger.error(f"❌ Video pipeline failed: {e}")
            raise

    def get_processor_summary(self) -> dict:
        """Get summary of processors in the pipeline."""
        return {
            "total_processors": len(self.processors),
            "processors": [
                {
                    "name": proc.__class__.__name__,
                    "type": type(proc).__name__,
                }
                for proc in self.processors
            ],
        }

    def add_processor(self, processor: Processor) -> None:
        """Add a custom processor to the pipeline."""
        self.processors.append(processor)
        self.logger.info(f"Added processor: {processor.__class__.__name__}")

    def remove_processor(self, processor_class) -> bool:
        """Remove a processor from the pipeline by class type."""
        for i, proc in enumerate(self.processors):
            if isinstance(proc, processor_class):
                removed = self.processors.pop(i)
                self.logger.info(f"Removed processor: {removed.__class__.__name__}")
                return True
        return False

    def clear_processors(self) -> None:
        """Clear all processors from the pipeline."""
        self.processors.clear()
        self.logger.info("Cleared all processors from pipeline")
