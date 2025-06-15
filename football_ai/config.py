"""
Football AI Configuration System

This module provides a comprehensive configuration system using dataclasses
for type-safe, flexible, and well-documented configuration management.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """Configuration for AI models and detection parameters."""

    # YOLO Detection Model Configuration
    player_model_path: str = "models/detect/best.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 1000
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # YOLO Keypoint Model Configuration
    field_model_path: str = "models/pose/best.pt"
    keypoint_confidence_threshold: float = 0.5
    keypoint_iou_threshold: float = 0.45
    enable_keypoint_detection: bool = True

    # Object Detection Classes
    player_class_ids: List[int] = field(default_factory=lambda: [0])
    ball_class_ids: List[int] = field(default_factory=lambda: [1])
    referee_class_ids: List[int] = field(default_factory=lambda: [2])
    goalkeeper_class_ids: List[int] = field(default_factory=lambda: [3])


@dataclass
class TrackingConfig:
    """Configuration for object tracking parameters."""

    # ByteTrack Parameters
    track_threshold: float = 0.6
    track_buffer: int = 30
    match_threshold: float = 0.8

    # Tracking Behavior
    min_track_length: int = 5
    max_lost_frames: int = 10
    track_smoothing_window: int = 3

    # Player-specific tracking
    player_track_threshold: float = 0.5
    ball_track_threshold: float = 0.3
    referee_track_threshold: float = 0.4


@dataclass
class TeamAnalysisConfig:
    """Configuration for team color analysis and assignment."""

    # Color Analysis
    n_clusters: int = 2
    color_samples: int = 100
    min_confidence: float = 0.6

    # Team Assignment
    assignment_history_length: int = 10
    color_stability_threshold: float = 0.7
    jersey_region_ratio: float = 0.6  # Upper portion of player for jersey analysis

    # Color Processing
    min_pixel_intensity: int = 30
    max_pixel_intensity: int = 220
    color_distance_threshold: float = 50.0

    # Timing Configuration
    enable_retry_analysis: bool = True
    max_analysis_attempts: int = 5
    min_players_for_analysis: int = 4
    analysis_start_frame: int = 30
    early_analysis_interval: int = 30  # Every 1 second at 30fps (frames 30-300)
    mid_analysis_interval: int = 150  # Every 5 seconds at 30fps (frames 300-900)
    late_analysis_interval: int = 300  # Every 10 seconds at 30fps (after frame 900)
    early_analysis_end_frame: int = 300
    mid_analysis_end_frame: int = 900


@dataclass
class PossessionConfig:
    """Configuration for ball possession analysis."""

    # Distance-based possession
    possession_distance: float = 70.0  # pixels
    confidence_threshold: int = 3
    history_length: int = 5

    # Temporal smoothing
    possession_smoothing_window: int = 7
    possession_stability_frames: int = 5

    # Ball-player association
    max_association_distance: float = 100.0
    association_confidence_threshold: float = 0.7


@dataclass
class CameraConfig:
    """Configuration for camera motion tracking."""

    # Feature Detection
    max_features: int = 100
    quality_level: float = 0.3
    min_distance: float = 3
    block_size: int = 7
    use_harris: bool = False

    # Optical Flow (when available)
    window_size: Tuple[int, int] = (15, 15)
    max_pyramid_level: int = 2

    # Movement Smoothing
    movement_smoothing_window: int = 5
    movement_threshold: float = 2.0  # minimum movement to track


@dataclass
class TransformationConfig:
    """Configuration for coordinate transformation."""

    # Field Dimensions (meters)
    field_width: float = 68.0
    field_height: float = 105.0

    # Keypoint Strategy Configuration
    keypoint_strategy: str = "manual"  # "manual", "auto", "hybrid"
    auto_detection_algorithm: str = (
        "line_detection"  # "line_detection", "deep_learning", "template_matching", "composite"
    )

    # Transformation Parameters
    auto_calibrate: bool = True
    calibration_confidence_threshold: float = 0.8

    # Default field keypoints (for manual strategy)
    default_keypoints: Optional[List[List[float]]] = field(
        default_factory=lambda: [[110, 1035], [265, 275], [910, 260], [1640, 915]]
    )

    # Auto-detection Parameters
    detection_retry_frames: int = 10  # Retry detection every N frames if failed
    detection_cache_duration: int = 100  # Cache successful detection for N frames

    # Coordinate System
    origin_position: str = "top_left"  # "top_left", "bottom_left", "center"
    coordinate_unit: str = "meters"  # "meters", "pixels", "normalized"


@dataclass
class RenderingConfig:
    """Configuration for video rendering and visualization."""

    # Display Options
    show_tracks: bool = True
    show_team_colors: bool = True
    show_ball_possession: bool = True
    show_speeds: bool = True
    show_positions: bool = True
    show_field_coordinates: bool = False
    show_keypoints: bool = False

    # Keypoint Visualization
    keypoint_radius: int = 3
    keypoint_thickness: int = 2
    show_pose_skeleton: bool = True
    skeleton_thickness: int = 2
    keypoint_confidence_threshold: float = 0.5

    # Track Visualization
    max_track_length: int = 30
    track_thickness: int = 2
    show_track_ids: bool = True

    # Annotation Styling
    bbox_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1

    # Colors (BGR format)
    colors: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            "team_1": (0, 255, 0),  # Green
            "team_2": (0, 0, 255),  # Blue
            "referee": (255, 255, 0),  # Yellow
            "ball": (255, 255, 255),  # White
            "unknown": (128, 128, 128),  # Gray
            "possession": (255, 0, 255),  # Magenta
            "track": (200, 200, 200),  # Light gray
            "text": (255, 255, 255),  # White
            "background": (0, 0, 0),  # Black
            "keypoint": (0, 255, 255),  # Cyan
            "skeleton": (255, 128, 0),  # Orange
        }
    )

    # Overlay Settings
    show_overlay_panel: bool = True
    overlay_transparency: float = 0.7
    overlay_position: str = (
        "top_left"  # "top_left", "top_right", "bottom_left", "bottom_right"
    )


@dataclass
class ProcessingConfig:
    """Configuration for video processing and performance."""

    # Video Processing
    input_video_path: str = ""
    output_video_path: str = ""
    output_directory: str = "outputs/data"

    # Performance Settings
    batch_size: int = 1
    num_workers: int = 4
    use_gpu_acceleration: bool = True

    # Caching
    enable_caching: bool = True
    cache_directory: str = "outputs/cache"
    load_from_cache: bool = False
    save_to_cache: bool = False

    # Processing Options
    process_every_nth_frame: int = 1
    max_frames_to_process: Optional[int] = None
    start_frame: int = 0

    # Output Formats
    save_annotated_video: bool = True
    save_analysis_data: bool = True
    save_statistics: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv", "pkl"])


@dataclass
class FootballAIConfig:
    """Main configuration class that combines all component configurations."""

    # Component Configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    team_analysis: TeamAnalysisConfig = field(default_factory=TeamAnalysisConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    transformation: TransformationConfig = field(default_factory=TransformationConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    # Global Settings
    debug_mode: bool = False
    verbose_logging: bool = True
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"

    # Validation Settings
    validate_inputs: bool = True
    strict_mode: bool = False

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, filepath: str) -> "FootballAIConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, "__dict__"):
                result[field_name] = field_value.__dict__
            else:
                result[field_name] = field_value
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FootballAIConfig":
        """Create configuration from dictionary."""
        # Extract component configs
        model_config = ModelConfig(**config_dict.get("model", {}))
        tracking_config = TrackingConfig(**config_dict.get("tracking", {}))
        team_analysis_config = TeamAnalysisConfig(
            **config_dict.get("team_analysis", {})
        )
        possession_config = PossessionConfig(**config_dict.get("possession", {}))
        camera_config = CameraConfig(**config_dict.get("camera", {}))
        transformation_config = TransformationConfig(
            **config_dict.get("transformation", {})
        )
        rendering_config = RenderingConfig(**config_dict.get("rendering", {}))
        processing_config = ProcessingConfig(**config_dict.get("processing", {}))

        # Create main config
        main_config_dict = {
            k: v
            for k, v in config_dict.items()
            if k
            not in [
                "model",
                "tracking",
                "team_analysis",
                "possession",
                "camera",
                "transformation",
                "rendering",
                "processing",
            ]
        }

        return cls(
            model=model_config,
            tracking=tracking_config,
            team_analysis=team_analysis_config,
            possession=possession_config,
            camera=camera_config,
            transformation=transformation_config,
            rendering=rendering_config,
            processing=processing_config,
            **main_config_dict,
        )

    def update_paths(
        self,
        model_path: Optional[str] = None,
        keypoint_model_path: Optional[str] = None,
        input_video_path: Optional[str] = None,
        output_video_path: Optional[str] = None,
        output_directory: Optional[str] = None,
    ) -> None:
        """Convenience method to update common paths."""
        if model_path:
            self.model.player_model_path = model_path
        if keypoint_model_path:
            self.model.field_model_path = keypoint_model_path
        if input_video_path:
            self.processing.input_video_path = input_video_path
        if output_video_path:
            self.processing.output_video_path = output_video_path
        if output_directory:
            self.processing.output_directory = output_directory

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate file paths
        if (
            self.model.player_model_path
            and not Path(self.model.player_model_path).exists()
        ):
            issues.append(f"Model file not found: {self.model.player_model_path}")

        if (
            self.model.enable_keypoint_detection
            and self.model.field_model_path
            and not Path(self.model.field_model_path).exists()
        ):
            issues.append(
                f"Keypoint model file not found: {self.model.field_model_path}"
            )

        if (
            self.processing.input_video_path
            and not Path(self.processing.input_video_path).exists()
        ):
            issues.append(f"Input video not found: {self.processing.input_video_path}")

        # Validate thresholds
        if not 0 <= self.model.confidence_threshold <= 1:
            issues.append("Model confidence threshold must be between 0 and 1")

        if not 0 <= self.model.iou_threshold <= 1:
            issues.append("Model IoU threshold must be between 0 and 1")

        if self.model.enable_keypoint_detection:
            if not 0 <= self.model.keypoint_confidence_threshold <= 1:
                issues.append(
                    "Keypoint model confidence threshold must be between 0 and 1"
                )

            if not 0 <= self.model.keypoint_iou_threshold <= 1:
                issues.append("Keypoint model IoU threshold must be between 0 and 1")

        # Validate field dimensions
        if (
            self.transformation.field_width <= 0
            or self.transformation.field_height <= 0
        ):
            issues.append("Field dimensions must be positive")

        return issues

    def get_summary(self) -> str:
        """Get a human-readable summary of the configuration."""
        summary = []
        summary.append("=== Football AI Configuration Summary ===")
        summary.append(f"Detection Model: {Path(self.model.player_model_path).name}")
        summary.append(
            f"Keypoint Model: {Path(self.model.field_model_path).name if self.model.enable_keypoint_detection else 'Disabled'}"
        )
        summary.append(
            f"Input Video: {Path(self.processing.input_video_path).name if self.processing.input_video_path else 'Not set'}"
        )
        summary.append(f"Output Directory: {self.processing.output_directory}")
        summary.append(f"Detection Confidence: {self.model.confidence_threshold}")
        summary.append(
            f"Keypoint Detection: {'Enabled' if self.model.enable_keypoint_detection else 'Disabled'}"
        )
        summary.append(f"Tracking Enabled: {self.tracking.track_threshold > 0}")
        summary.append(f"Team Analysis: {self.team_analysis.n_clusters} teams")
        summary.append(
            f"Ball Possession: Distance threshold {self.possession.possession_distance}px"
        )
        summary.append(f"Camera Tracking: {self.camera.max_features} features")
        summary.append(
            f"Field Size: {self.transformation.field_width}x{self.transformation.field_height}m"
        )
        summary.append(
            f"Rendering: {'Enabled' if self.rendering.show_tracks else 'Disabled'}"
        )
        summary.append(
            f"Caching: {'Enabled' if self.processing.enable_caching else 'Disabled'}"
        )
        summary.append(f"Debug Mode: {'On' if self.debug_mode else 'Off'}")

        return "\n".join(summary)


# Predefined configuration presets
def get_default_config() -> FootballAIConfig:
    """Get default configuration for general use."""
    config = FootballAIConfig()
    # Set default model paths to absolute paths if they exist
    default_model_path = "/workspaces/football_analysis/models/detect/best.pt"
    if Path(default_model_path).exists():
        config.model.player_model_path = default_model_path

    default_keypoint_model_path = "/workspaces/football_analysis/models/pose/best.pt"
    if Path(default_keypoint_model_path).exists():
        config.model.field_model_path = default_keypoint_model_path

    return config


def get_high_accuracy_config() -> FootballAIConfig:
    """Get configuration optimized for high accuracy analysis."""
    config = FootballAIConfig()

    # Higher detection thresholds
    config.model.confidence_threshold = 0.7
    config.tracking.track_threshold = 0.7

    # More stable tracking
    config.tracking.track_buffer = 50
    config.tracking.track_smoothing_window = 5

    # Better team analysis
    config.team_analysis.color_samples = 200
    config.team_analysis.min_confidence = 0.8

    # More reliable possession detection
    config.possession.confidence_threshold = 5
    config.possession.possession_smoothing_window = 10

    return config


def get_fast_processing_config() -> FootballAIConfig:
    """Get configuration optimized for fast processing."""
    config = FootballAIConfig()

    # Lower thresholds for speed
    config.model.confidence_threshold = 0.4
    config.model.max_detections = 500

    # Reduced tracking complexity
    config.tracking.track_buffer = 15
    config.camera.max_features = 50

    # Simplified rendering
    config.rendering.show_tracks = False
    config.rendering.show_field_coordinates = False
    config.rendering.max_track_length = 15

    # Process every 2nd frame
    config.processing.process_every_nth_frame = 2

    return config


def get_broadcast_config() -> FootballAIConfig:
    """Get configuration optimized for broadcast-quality output."""
    config = FootballAIConfig()

    # High-quality rendering
    config.rendering.show_tracks = True
    config.rendering.show_team_colors = True
    config.rendering.show_ball_possession = True
    config.rendering.show_speeds = True
    config.rendering.max_track_length = 50
    config.rendering.bbox_thickness = 3
    config.rendering.font_scale = 0.7

    # Professional styling
    config.rendering.show_overlay_panel = True
    config.rendering.overlay_transparency = 0.8

    # High accuracy
    config.model.confidence_threshold = 0.6
    config.tracking.track_threshold = 0.6

    return config
