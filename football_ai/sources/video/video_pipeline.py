"""
Video Analysis Pipeline - Orchestrates all video processing components.

This module contains the video analysis pipeline that processes videos
using all the detection, tracking, and analysis processors.
"""

from typing import Optional, List
from pathlib import Path

from ...config import FootballAIConfig, get_default_config
from ...domain.video import Video
from ...domain.interfaces import Processor
from ...utils import setup_logger, create_progress_bar

# Import all video analysis processors
from .processors.detection.yolo_detector import ObjectDetectionProcessor
from .processors.tracking.byte_tracker import TrackProcessor
from .processors.motion.speed_calculator import ObjectMotionProcessor
from .processors.motion.camera_stabilizer import CameraMotionProcessor
from .processors.transformation.coordinate_transformer import (
    FieldTransformationProcessor,
)
from .processors.assignment.siglip_team_classifier import (
    SigLIPTeamAssignmentProcessor,
)
from .processors.assignment.possession_analyzer import BallAssignmentProcessor
from .processors.analysis.speed_analyzer import SpeedProcessor
from .processors.analysis.possession_tracker import BallControlProcessor
from .processors.rendering.video_annotator import RendererProcessor
from .processors.storing.video_exporter import VideoWriterProcessor


class VideoPipeline:
    """
    Video analysis pipeline that processes videos through all analysis stages.

    This class orchestrates the video processing workflow and provides
    video analysis capabilities within the game-centric architecture.
    """

    def __init__(self, config: Optional[FootballAIConfig] = None):
        self.config = config or get_default_config()
        self.logger = setup_logger("VideoPipeline", self.config.log_level)
        self.processors: List[Processor] = []
        self._build_processors()

    def _build_processors(self) -> None:
        """Build the video processing pipeline based on configuration."""
        self.processors = []

        # Core detection and tracking
        self.processors.append(
            ObjectDetectionProcessor(self.config.model.player_model_path)
        )

        self.processors.append(
            TrackProcessor(
                track_activation_threshold=self.config.tracking.track_activation_threshold,
                lost_track_buffer=self.config.tracking.lost_track_buffer,
                minimum_matching_threshold=self.config.tracking.minimum_matching_threshold,
                frame_rate=self.config.tracking.frame_rate,
                minimum_consecutive_frames=self.config.tracking.minimum_consecutive_frames,
                min_track_length=self.config.tracking.min_track_length,
            )
        )

        # Motion analysis
        self.processors.append(ObjectMotionProcessor())
        self.processors.append(CameraMotionProcessor())

        # Field transformation (if configured)
        if hasattr(self.config, "field_corners") and self.config.field_corners:
            self.processors.append(
                FieldTransformationProcessor(pixel_corners=self.config.field_corners)
            )

        # Team and ball assignment
        team_assignment_processor = SigLIPTeamAssignmentProcessor(
            model_path=self.config.model.team_model_path,
            batch_size=self.config.model.team_batch_size,
        )
        self.processors.append(team_assignment_processor)

        self.processors.append(
            BallAssignmentProcessor(
                max_distance=self.config.possession.possession_distance
            )
        )

        self.processors.append(BallControlProcessor())

        # Analysis
        self.processors.append(SpeedProcessor())

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
            # Validate configuration if in strict mode
            if self.config.strict_mode:
                issues = self.config.validate()
                if issues:
                    raise ValueError(f"Configuration issues: {issues}")

            self.logger.info("Starting video analysis pipeline")

            # Process through pipeline
            progress_bar = create_progress_bar(
                iterable=self.processors,
                desc="Processing video pipeline",
                unit="processor",
                disable=not self.config.show_progress_bars,
            )

            for i, processor in enumerate(progress_bar):
                processor_name = processor.__class__.__name__
                progress_bar.set_description(f"Running {processor_name}")

                self.logger.info(
                    f"Running processor {i+1}/{len(self.processors)}: {processor_name}"
                )

                try:
                    video = processor.process(video)
                    self.logger.debug(f"✓ {processor_name} completed successfully")
                except Exception as e:
                    if self.config.strict_mode:
                        progress_bar.close()
                        raise RuntimeError(
                            f"Processor {processor_name} failed: {e}"
                        ) from e
                    else:
                        self.logger.error(f"⚠ {processor_name} failed: {e}")
                        continue

            progress_bar.close()

            # Add rendering and output if specified
            if output_path:
                self._add_output_processing(video, output_path)

            self.logger.info("✅ Video analysis pipeline completed successfully")
            return video

        except Exception as e:
            self.logger.error(f"❌ Video pipeline failed: {e}")
            raise

    def _add_output_processing(self, video: Video, output_path: str) -> Video:
        """Add rendering and video writing to produce output video."""
        try:
            # Add renderer
            renderer = RendererProcessor(
                output_path=output_path,
                render_config=self.config.rendering.__dict__,
            )
            video = renderer.process(video)

            # Add video writer
            writer = VideoWriterProcessor(output_path=output_path)
            video = writer.process(video)

            self.logger.info(f"Output video saved to: {output_path}")
            return video

        except Exception as e:
            self.logger.error(f"Output processing failed: {e}")
            # Don't fail the whole pipeline for output issues
            return video

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
            "config_summary": {
                "strict_mode": self.config.strict_mode,
                "debug_mode": self.config.debug_mode,
                "show_progress": self.config.show_progress_bars,
            },
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
