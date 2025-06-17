"""
Main Football Analysis Pipeline

This module provides the main orchestrator class that coordinates all processors
to perform comprehensive football video analysis.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
from tqdm import tqdm

from .config import FootballAIConfig, get_default_config
from .domain.data_models import VideoData, FrameData
from .domain.interfaces import Processor

# Import all processors
from .detection.object_detection_processor import ObjectDetectionProcessor
from .tracking.track_processor import TrackProcessor
from .motion.object_motion_processor import ObjectMotionProcessor
from .motion.camera_motion_processor import CameraMotionProcessor
from .transformation.field_transformation_processor import FieldTransformationProcessor
from .assignment.team_assignment_processor import TeamAssignmentProcessor
from .assignment.ball_assignment_processor import BallAssignmentProcessor
from .analysis.speed_processor import SpeedProcessor
from .rendering.renderer_processor import RendererProcessor
from .storing.video_writer_processor import VideoWriterProcessor


class FootballAnalysisPipeline:
    """
    Main pipeline for football video analysis.

    Orchestrates all processors to perform comprehensive analysis including:
    - Object detection and tracking
    - Team assignment and ball possession
    - Speed and motion analysis
    - Video rendering and output
    """

    def __init__(self, config: Optional[FootballAIConfig] = None):
        """
        Initialize the football analysis pipeline.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or get_default_config()
        self.logger = self._setup_logging()
        self.processors: List[Processor] = []
        self._build_processors()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("FootballAnalysisPipeline")
        logger.setLevel(getattr(logging, self.config.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _build_processors(self) -> None:
        """Build the processor pipeline based on configuration."""
        self.processors = []

        # Core detection and tracking
        self.processors.append(
            ObjectDetectionProcessor(self.config.model.player_model_path)
        )
        self.processors.append(TrackProcessor())

        # Motion and positioning
        self.processors.append(ObjectMotionProcessor())
        if self.config.camera.enable_camera_tracking:
            self.processors.append(CameraMotionProcessor())

        # Field transformation
        self.processors.append(
            FieldTransformationProcessor(
                field_width=self.config.transformation.field_width,
                field_height=self.config.transformation.field_height,
            )
        )

        # Team and ball assignment
        self.processors.append(TeamAssignmentProcessor())
        self.processors.append(
            BallAssignmentProcessor(
                max_distance=self.config.possession.possession_distance
            )
        )

        # Analysis
        self.processors.append(SpeedProcessor())

        # Rendering (if output video requested)
        if (
            hasattr(self.config.processing, "output_video_path")
            and self.config.processing.output_video_path
        ):
            # Add renderer to annotate frames
            self.processors.append(
                RendererProcessor(
                    output_path=self.config.processing.output_video_path,
                    render_config=self.config.rendering.__dict__,
                )
            )
            # Add video writer to save the annotated frames
            self.processors.append(
                VideoWriterProcessor(
                    output_path=self.config.processing.output_video_path
                )
            )

        self.logger.info(f"Built pipeline with {len(self.processors)} processors")

    def load_video(self, video_path: str) -> VideoData:
        """
        Load video from file and create VideoData object.

        Args:
            video_path: Path to input video file

        Returns:
            VideoData object with loaded frames
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        # Get video properties
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / frame_rate if frame_rate > 0 else 0

        self.logger.info(f"Loading video: {video_path}")
        self.logger.info(
            f"Properties: {width}x{height} @ {frame_rate}fps, {total_frames} frames"
        )

        frames = []
        frame_number = 0

        # Load frames based on processing configuration
        process_every_nth = getattr(
            self.config.processing, "process_every_nth_frame", 1
        )

        # Use tqdm for progress bar when loading frames
        progress_bar = tqdm(
            total=total_frames,
            desc="Loading video frames",
            unit="frames",
            disable=not self.config.show_progress_bars,
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % process_every_nth == 0:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frames.append(
                    FrameData(
                        frame_number=frame_number, timestamp=timestamp, raw_frame=frame
                    )
                )

            frame_number += 1
            progress_bar.update(1)

        progress_bar.close()
        cap.release()

        video_data = VideoData(
            video_path=video_path,
            frame_rate=frame_rate,
            resolution=(width, height),
            duration=duration,
            frames=frames,
        )

        self.logger.info(f"Loaded {len(frames)} frames for processing")
        return video_data

    def process_video(
        self, video_path: str, output_path: Optional[str] = None
    ) -> VideoData:
        """
        Process a video through the complete analysis pipeline.

        Args:
            video_path: Path to input video file
            output_path: Optional path for output video

        Returns:
            VideoData with complete analysis results
        """
        try:
            # Validate configuration
            issues = self.config.validate()
            if issues and self.config.strict_mode:
                raise ValueError(f"Configuration issues: {issues}")
            elif issues:
                self.logger.warning(f"Configuration warnings: {issues}")

            # Load video
            video_data = self.load_video(video_path)

            # Process through pipeline
            self.logger.info("Starting video analysis pipeline")

            # Use progress bar for processor execution
            processor_progress = tqdm(
                self.processors,
                desc="Processing pipeline",
                unit="processor",
                disable=not self.config.show_progress_bars,
            )

            for i, processor in enumerate(processor_progress):
                processor_name = processor.__class__.__name__
                processor_progress.set_description(f"Running {processor_name}")
                self.logger.info(
                    f"Running processor {i+1}/{len(self.processors)}: {processor_name}"
                )

                try:
                    video_data = processor.process(video_data)
                    self.logger.debug(f"✓ {processor_name} completed successfully")

                except Exception as e:
                    if self.config.strict_mode:
                        processor_progress.close()
                        raise RuntimeError(
                            f"Processor {processor_name} failed: {e}"
                        ) from e
                    else:
                        self.logger.error(f"⚠ {processor_name} failed: {e}")
                        continue

            processor_progress.close()

            # Save output video if writer processor wasn't included
            if output_path and not any(
                isinstance(p, (RendererProcessor, VideoWriterProcessor))
                for p in self.processors
            ):
                self.logger.info(f"Saving output video to: {output_path}")
                writer = VideoWriterProcessor(output_path)
                video_data = writer.process(video_data)

            self.logger.info("✅ Video analysis pipeline completed successfully")
            return video_data

        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            raise

    def get_analysis_summary(self, video_data: VideoData) -> Dict[str, Any]:
        """
        Generate a summary of analysis results.

        Args:
            video_data: Processed video data

        Returns:
            Dictionary with analysis summary
        """
        if not video_data.frames:
            return {"error": "No frames to analyze"}

        # Count detections by type
        detection_counts = {"player": 0, "goalkeeper": 0, "referee": 0, "ball": 0}
        tracked_objects = set()
        team_assignments = {"team_1": 0, "team_2": 0, "unknown": 0}

        for frame_data in video_data.frames:
            for detection in frame_data.detections or []:
                # Count by type
                obj_type = detection.object_type or "unknown"
                if obj_type in detection_counts:
                    detection_counts[obj_type] += 1

                # Track unique objects
                track_id = (
                    detection.metadata.get("track_id") if detection.metadata else None
                )
                if track_id is not None:
                    tracked_objects.add(track_id)

                # Count team assignments
                team = detection.metadata.get("team") if detection.metadata else None
                if team == 1:
                    team_assignments["team_1"] += 1
                elif team == 2:
                    team_assignments["team_2"] += 1
                else:
                    team_assignments["unknown"] += 1

        summary = {
            "video_info": {
                "path": video_data.video_path,
                "duration": video_data.duration,
                "frames_processed": len(video_data.frames),
                "resolution": video_data.resolution,
                "frame_rate": video_data.frame_rate,
            },
            "detection_summary": detection_counts,
            "tracking_summary": {
                "unique_tracks": len(tracked_objects),
                "total_detections": sum(detection_counts.values()),
            },
            "team_summary": team_assignments,
            "pipeline_config": {
                "processors_used": len(self.processors),
                "strict_mode": self.config.strict_mode,
                "debug_mode": self.config.debug_mode,
            },
        }

        return summary

    def save_results(self, video_data: VideoData, output_path: str) -> None:
        """
        Save analysis results to file.

        Args:
            video_data: Processed video data
            output_path: Path to save results
        """
        import pickle

        results = {
            "summary": self.get_analysis_summary(video_data),
            "config": self.config.to_dict(),
            "video_data": video_data,
        }

        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        self.logger.info(f"Results saved to: {output_path}")

    def _process_through_pipeline(self, video_data: VideoData) -> VideoData:
        """
        Internal method to process video data through all processors.
        Used for testing and internal processing.

        Args:
            video_data: Video data to process

        Returns:
            Processed video data
        """
        for i, processor in enumerate(self.processors):
            processor_name = processor.__class__.__name__
            self.logger.debug(f"Processing with {processor_name}")

            try:
                video_data = processor.process(video_data)
                self.logger.debug(f"✓ {processor_name} completed successfully")

            except Exception as e:
                if self.config.strict_mode:
                    raise RuntimeError(f"Processor {processor_name} failed: {e}") from e
                else:
                    self.logger.error(f"⚠ {processor_name} failed: {e}")
                    continue

        return video_data
