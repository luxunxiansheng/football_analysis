from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import numpy as np
from pathlib import Path
import time

# Import specialized domain models
from .frame import Frame
from .field import Field


@dataclass
class VideoMetadata:
    """Video file metadata and technical information."""

    # File information
    file_path: str
    file_size: Optional[int] = None
    codec: Optional[str] = None
    container_format: Optional[str] = None
    bitrate: Optional[int] = None

    # Video properties
    frame_rate: float = 30.0
    resolution: Tuple[int, int] = (1920, 1080)  # (width, height)
    duration: float = 0.0
    total_frames: Optional[int] = None
    aspect_ratio: Optional[float] = None

    # Quality metrics
    average_quality: Optional[float] = None
    quality_variance: Optional[float] = None
    compression_ratio: Optional[float] = None


@dataclass
class MatchContext:
    """Match context and metadata."""

    # Match identification
    match_id: Optional[str] = None
    match_date: Optional[str] = None
    competition: Optional[str] = None
    season: Optional[str] = None
    round: Optional[str] = None

    # Teams and venue
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    home_team_id: Optional[str] = None
    away_team_id: Optional[str] = None
    venue: Optional[str] = None
    stadium_capacity: Optional[int] = None

    # Match details
    referee: Optional[str] = None
    weather: Optional[str] = None
    temperature: Optional[float] = None
    attendance: Optional[int] = None

    # Score and events
    final_score: Optional[Tuple[int, int]] = None  # (home, away)
    half_time_score: Optional[Tuple[int, int]] = None
    match_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProcessingConfig:
    """Processing configuration and pipeline settings."""

    # Detection settings
    detection_model: Optional[str] = None
    detection_confidence: float = 0.5
    detection_iou: float = 0.5

    # Tracking settings
    tracking_method: Optional[str] = None
    tracking_max_age: int = 30
    tracking_min_hits: int = 3

    # Field detection settings
    field_detection_method: Optional[str] = None
    field_calibration_method: Optional[str] = None

    # Analysis settings
    analyze_formations: bool = True
    analyze_possession: bool = True
    analyze_movement: bool = True

    # Output settings
    save_intermediate_results: bool = False
    output_format: str = "json"

    # Performance settings
    batch_size: int = 1
    num_workers: int = 1
    gpu_enabled: bool = False


@dataclass
class Video:
    """
    Comprehensive video model that consolidates all video-related data and functionality.
    This replaces VideoData with a more feature-rich, self-contained class.
    """

    # Core video identification
    video_id: str
    video_path: str

    # Video metadata and technical info
    metadata: VideoMetadata = field(default_factory=lambda: VideoMetadata(""))
    match_context: MatchContext = field(default_factory=MatchContext)
    processing_config: ProcessingConfig = field(default_factory=ProcessingConfig)

    # Frame collection
    frames: List[Frame] = field(default_factory=list)

    # Global field configuration (shared across frames if consistent)
    global_field: Optional[Field] = None

    # Processing state and results
    processing_start_time: Optional[float] = None
    processing_end_time: Optional[float] = None
    processing_errors: List[str] = field(default_factory=list)

    # Analysis results (video-level)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    # Export and output paths
    output_directory: Optional[str] = None
    exported_files: Dict[str, str] = field(default_factory=dict)  # type -> path

    # Additional custom data
    custom: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)

    # Explicit ball control statistics for match analysis
    ball_control_stats: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize video-specific data after creation."""
        if not self.metadata.file_path:
            self.metadata.file_path = self.video_path

        # Set video ID from path if not provided
        if not self.video_id and self.video_path:
            self.video_id = Path(self.video_path).stem

    # Frame management methods
    def add_frame(self, frame: Frame) -> None:
        """Add a frame to the video."""
        if frame.frame_number in [f.frame_number for f in self.frames]:
            raise ValueError(f"Frame {frame.frame_number} already exists")

        # Keep frames sorted by frame number
        self.frames.append(frame)
        self.frames.sort(key=lambda f: f.frame_number)

    def insert_frame(self, frame: Frame, index: int) -> None:
        """Insert a frame at a specific index."""
        self.frames.insert(index, frame)

    def remove_frame(self, frame_number: int) -> Optional[Frame]:
        """Remove and return a frame by frame number."""
        for i, frame in enumerate(self.frames):
            if frame.frame_number == frame_number:
                return self.frames.pop(i)
        return None

    def get_frame(self, frame_number: int) -> Optional[Frame]:
        """Get a specific frame by frame number."""
        for frame in self.frames:
            if frame.frame_number == frame_number:
                return frame
        return None

    def get_frame_by_timestamp(
        self, timestamp: float, tolerance: float = 0.1
    ) -> Optional[Frame]:
        """Get a frame by timestamp with tolerance."""
        for frame in self.frames:
            if abs(frame.timestamp - timestamp) <= tolerance:
                return frame
        return None

    def get_frames_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[Frame]:
        """Get frames within a time range."""
        return [
            frame for frame in self.frames if start_time <= frame.timestamp <= end_time
        ]

    def get_frames_by_match_time_range(
        self, start_match_time: float, end_match_time: float
    ) -> List[Frame]:
        """Get frames within a match time range."""
        return [
            frame
            for frame in self.frames
            if frame.match_time
            and start_match_time <= frame.match_time <= end_match_time
        ]

    def get_frames_by_period(self, period: str) -> List[Frame]:
        """Get frames for a specific match period."""
        return [frame for frame in self.frames if frame.match_period == period]

    # Field management
    def set_global_field(self, field: Field) -> None:
        """Set a global field configuration for all frames."""
        self.global_field = field
        for frame in self.frames:
            frame.field = field

    def calibrate_field_for_all_frames(
        self,
        corners: List[Tuple[float, float]],
        confidence: float,
        method: str = "manual",
    ) -> None:
        """Calibrate field for all frames."""
        if self.global_field:
            self.global_field.set_calibration(corners, confidence, method)

        for frame in self.frames:
            frame.set_field_calibration(corners, confidence, method)

    def is_field_calibrated(self) -> bool:
        """Check if field is calibrated."""
        if self.global_field:
            return self.global_field.is_calibrated
        return any(frame.field.is_calibrated for frame in self.frames if frame.field)

    # Analysis methods
    def get_total_objects_detected(self) -> Dict[str, int]:
        """Get total count of all objects detected across all frames."""
        totals = {"players": 0, "goalkeepers": 0, "referees": 0, "ball": 0}

        for frame in self.frames:
            counts = frame.get_active_objects_count()
            for obj_type in totals:
                totals[obj_type] += counts.get(obj_type, 0)

        totals["total"] = sum(totals.values())
        return totals

    def get_average_objects_per_frame(self) -> Dict[str, float]:
        """Get average number of objects per frame."""
        if not self.frames:
            return {"players": 0.0, "goalkeepers": 0.0, "referees": 0.0, "ball": 0.0}

        totals = self.get_total_objects_detected()
        frame_count = len(self.frames)

        return {
            obj_type: count / frame_count
            for obj_type, count in totals.items()
            if obj_type != "total"
        }

    def analyze_possession_by_period(self) -> Dict[str, Dict[str, float]]:
        """Analyze ball possession by match period."""
        periods = {}

        for period in ["first_half", "second_half", "extra_time"]:
            period_frames = self.get_frames_by_period(period)
            if not period_frames:
                continue

            team_possession = {}
            total_frames = len(period_frames)

            for frame in period_frames:
                ball_info = frame.get_ball_control_info()
                if ball_info and ball_info.get("possession_team"):
                    team_id = ball_info["possession_team"]
                    team_possession[team_id] = team_possession.get(team_id, 0) + 1

            # Convert to percentages
            periods[period] = {
                str(team_id): (count / total_frames) * 100
                for team_id, count in team_possession.items()
            }

        return periods

    def analyze_formation_changes(self, team_id: int) -> List[Dict[str, Any]]:
        """Analyze formation changes for a team throughout the match."""
        formations = []
        current_formation = None

        for frame in self.frames:
            formation = frame.get_formation_string(team_id)

            if formation != current_formation:
                formations.append(
                    {
                        "timestamp": frame.timestamp,
                        "match_time": frame.match_time,
                        "frame_number": frame.frame_number,
                        "formation": formation,
                        "previous_formation": current_formation,
                    }
                )
                current_formation = formation

        return formations

    def get_heatmap_data(
        self, team_id: Optional[int] = None
    ) -> Dict[int, List[Tuple[float, float]]]:
        """Get heatmap data for players."""
        heatmap_data = {}

        for frame in self.frames:
            players = frame.get_all_players()

            for player in players:
                if team_id is not None and player.team_id != team_id:
                    continue

                if player.track_id not in heatmap_data:
                    heatmap_data[player.track_id] = []

                if player.field_position:
                    heatmap_data[player.track_id].append(player.field_position)

        return heatmap_data

    # Processing management
    def start_processing(self) -> None:
        """Mark the start of processing."""
        self.processing_start_time = time.time()
        self.processing_errors.clear()

    def finish_processing(self) -> None:
        """Mark the end of processing."""
        self.processing_end_time = time.time()

    def get_processing_time(self) -> Optional[float]:
        """Get total processing time."""
        if self.processing_start_time and self.processing_end_time:
            return self.processing_end_time - self.processing_start_time
        return None

    def add_processing_error(self, error: str) -> None:
        """Add a processing error."""
        self.processing_errors.append(error)

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary."""
        processed_frames = 0
        frames_with_errors = 0
        total_frame_processing_time = 0.0

        for frame in self.frames:
            if frame.is_processing_complete():
                processed_frames += 1
            if frame.has_errors():
                frames_with_errors += 1
            if frame.processing_time:
                total_frame_processing_time += frame.processing_time

        return {
            "total_frames": len(self.frames),
            "processed_frames": processed_frames,
            "frames_with_errors": frames_with_errors,
            "processing_complete": processed_frames == len(self.frames),
            "total_processing_time": self.get_processing_time(),
            "frame_processing_time": total_frame_processing_time,
            "avg_frame_processing_time": (
                total_frame_processing_time / len(self.frames) if self.frames else 0
            ),
            "processing_errors": len(self.processing_errors),
            "frame_errors": frames_with_errors,
        }

    # Export and output methods
    def set_output_directory(self, directory: str) -> None:
        """Set the output directory for exports."""
        self.output_directory = directory
        Path(directory).mkdir(parents=True, exist_ok=True)

    def export_frame_data(self, format: str = "json") -> str:
        """Export frame data to file."""
        if not self.output_directory:
            raise ValueError("Output directory not set")

        filename = f"{self.video_id}_frames.{format}"
        filepath = str(Path(self.output_directory) / filename)

        frame_data = [frame.get_frame_info() for frame in self.frames]

        if format == "json":
            import json

            with open(filepath, "w") as f:
                json.dump(frame_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.exported_files["frames"] = filepath
        return filepath

    def export_analysis_results(self, format: str = "json") -> str:
        """Export analysis results to file."""
        if not self.output_directory:
            raise ValueError("Output directory not set")

        filename = f"{self.video_id}_analysis.{format}"
        filepath = str(Path(self.output_directory) / filename)

        analysis_data = {
            "video_info": self.get_video_info(),
            "possession_analysis": self.analyze_possession_by_period(),
            "objects_detected": self.get_total_objects_detected(),
            "average_objects": self.get_average_objects_per_frame(),
            "processing_summary": self.get_processing_summary(),
            "custom_analysis": self.analysis_results,
        }

        if format == "json":
            import json

            with open(filepath, "w") as f:
                json.dump(analysis_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.exported_files["analysis"] = filepath
        return filepath

    # Information and metadata methods
    def get_video_info(self) -> Dict[str, Any]:
        """Get comprehensive video information."""
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "metadata": {
                "file_size": self.metadata.file_size,
                "codec": self.metadata.codec,
                "container_format": self.metadata.container_format,
                "bitrate": self.metadata.bitrate,
                "frame_rate": self.metadata.frame_rate,
                "resolution": self.metadata.resolution,
                "duration": self.metadata.duration,
                "total_frames": self.metadata.total_frames,
                "aspect_ratio": self.metadata.aspect_ratio,
            },
            "match_context": {
                "match_id": self.match_context.match_id,
                "match_date": self.match_context.match_date,
                "competition": self.match_context.competition,
                "home_team": self.match_context.home_team,
                "away_team": self.match_context.away_team,
                "venue": self.match_context.venue,
                "final_score": self.match_context.final_score,
            },
            "frames_loaded": len(self.frames),
            "field_calibrated": self.is_field_calibrated(),
            "processing_complete": self.get_processing_summary()["processing_complete"],
            "objects_detected": self.get_total_objects_detected(),
            "tags": self.tags,
        }

    def get_technical_specs(self) -> Dict[str, Any]:
        """Get technical video specifications."""
        return {
            "file_path": self.video_path,
            "file_size": self.metadata.file_size,
            "codec": self.metadata.codec,
            "container_format": self.metadata.container_format,
            "bitrate": self.metadata.bitrate,
            "frame_rate": self.metadata.frame_rate,
            "resolution": self.metadata.resolution,
            "duration": self.metadata.duration,
            "total_frames": self.metadata.total_frames,
            "aspect_ratio": self.metadata.aspect_ratio,
            "quality_metrics": self.quality_metrics,
        }

    # Utility methods
    def clone(self, include_frames: bool = True) -> "Video":
        """Create a copy of this video, optionally including frames."""
        new_video = Video(
            video_id=f"{self.video_id}_copy",
            video_path=self.video_path,
        )

        # Copy metadata
        new_video.metadata = VideoMetadata(**self.metadata.__dict__)
        new_video.match_context = MatchContext(**self.match_context.__dict__)
        new_video.processing_config = ProcessingConfig(
            **self.processing_config.__dict__
        )

        # Copy other attributes
        new_video.global_field = self.global_field
        new_video.processing_start_time = self.processing_start_time
        new_video.processing_end_time = self.processing_end_time
        new_video.processing_errors = self.processing_errors.copy()
        new_video.analysis_results = self.analysis_results.copy()
        new_video.quality_metrics = self.quality_metrics.copy()
        new_video.output_directory = self.output_directory
        new_video.exported_files = self.exported_files.copy()
        new_video.custom = self.custom.copy()
        new_video.tags = self.tags.copy()
        new_video.annotations = self.annotations.copy()

        # Optionally copy frames
        if include_frames:
            new_video.frames = [frame.clone() for frame in self.frames]

        return new_video

    def add_tag(self, tag: str) -> None:
        """Add a tag to the video."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from the video."""
        if tag in self.tags:
            self.tags.remove(tag)
            return True
        return False

    def has_tag(self, tag: str) -> bool:
        """Check if video has a specific tag."""
        return tag in self.tags

    # Iterator protocol
    def __len__(self) -> int:
        """Return number of frames."""
        return len(self.frames)

    def __getitem__(self, index: int) -> Frame:
        """Get frame by index."""
        return self.frames[index]

    def __iter__(self) -> Iterator[Frame]:
        """Iterate over frames."""
        return iter(self.frames)

    def __str__(self) -> str:
        """String representation of the video."""
        return f"Video({self.video_id}, {len(self.frames)} frames, {self.metadata.duration:.1f}s)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Video(id='{self.video_id}', path='{self.video_path}', "
            f"frames={len(self.frames)}, duration={self.metadata.duration:.1f}s, "
            f"resolution={self.metadata.resolution})"
        )


# Keep the old VideoData as an alias for backward compatibility during transition
VideoData = Video
