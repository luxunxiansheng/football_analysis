"""
Modern Football Analysis Pipeline

This module provides the main pipeline that integrates all components
to replicate the functionality of the original notebook workflow.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import pickle
import os
import contextlib
from tqdm import tqdm

from .config import FootballAIConfig, get_default_config
from .domain.models import (
    Detection,
    PlayerState,
    ObjectType,
    TeamAssignment,
    MatchAnalysis,
)
from .domain.interfaces import FieldKeypointDetector
from .detection.yolo_detector import YOLODetector
from .tracking.byte_tracker import ByteTracker
from .analysis.team_color_analyzer import KMeansTeamColorAnalyzer
from .analysis.ball_possession_analyzer import DistanceBasedBallPossessionAnalyzer
from .motion.camera_motion_tracker import OpticalFlowCameraTracker
from .transformation.coordinate_transformer import PerspectiveCoordinateTransformer
from .rendering.video_renderer import VideoRenderer
from .utils.video_utils import read_video_frames, get_video_properties, VideoWriter


class FootballAnalysisPipeline:
    """
    Modern football analysis pipeline that replicates the original
    notebook functionality with clean architecture.

    This pipeline provides clean, high-level API for football video analysis.
    Implementation details like coordinate transformation calibration are handled
    internally through the configuration system.

    Example Usage:
        # Basic usage
        pipeline = FootballAnalysisPipeline()
        results = pipeline.process_video("match.mp4")

        # With field calibration
        pipeline = FootballAnalysisPipeline()
        pipeline.set_field_keypoints([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
        results = pipeline.process_video("match.mp4", "output.mp4")

        # With custom config
        config = get_default_config()
        config.transformation.default_keypoints = field_corners
        pipeline = FootballAnalysisPipeline(config=config)
        results = pipeline.process_video("match.mp4")
    """

    def __init__(
        self,
        config: Optional[FootballAIConfig] = None,
        model_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        save_cache: Optional[bool] = None,
        load_cache: Optional[bool] = None,
    ):
        """
        Initialize the football analysis pipeline.

        Args:
            config: Football AI configuration object. If None, uses default config.
            model_path: Path to YOLO model file (overrides config if provided)
            output_dir: Directory for output files (overrides config if provided)
            save_cache: Whether to save processing cache (overrides config if provided)
            load_cache: Whether to load from cache if available (overrides config if provided)
        """
        # Initialize configuration
        self.config = config if config is not None else get_default_config()

        # Override config with any explicitly provided parameters
        if model_path is not None:
            self.config.model.model_path = model_path
        if output_dir is not None:
            self.config.processing.output_directory = output_dir
        if save_cache is not None:
            self.config.processing.save_to_cache = save_cache
        if load_cache is not None:
            self.config.processing.load_from_cache = load_cache

        # Validate configuration
        validation_issues = self.config.validate()
        if validation_issues:
            print("Configuration validation warnings:")
            for issue in validation_issues:
                print(f"  - {issue}")

        # Create output directory
        os.makedirs(self.config.processing.output_directory, exist_ok=True)

        # Initialize components with configuration
        self.detector = YOLODetector(
            model_path=self.config.model.model_path,
            confidence_threshold=self.config.model.confidence_threshold,
        )

        self.tracker = ByteTracker()

        self.team_analyzer = KMeansTeamColorAnalyzer()

        self.possession_analyzer = DistanceBasedBallPossessionAnalyzer()

        self.camera_tracker = OpticalFlowCameraTracker()

        # Initialize coordinate transformer with field corners from config
        self.coordinate_transformer = PerspectiveCoordinateTransformer(
            field_corners=self.config.transformation.default_keypoints
        )

        self.renderer = VideoRenderer()

        # Analysis results
        self.analysis_results: Optional[MatchAnalysis] = None

        # Processing state
        self.is_initialized = False

        if self.config.verbose_logging:
            print("Football AI analysis pipeline initialized successfully!")
            if self.config.debug_mode:
                print(self.config.get_summary())

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
    ) -> MatchAnalysis:
        """
        Process a football video and generate complete analysis.

        Args:
            video_path: Path to input video
            output_video_path: Path for annotated output video

        Returns:
            Complete match analysis results
        """
        print(f"Starting analysis of video: {video_path}")

        # Try to load cached results first
        cached_results = self._try_load_cache()
        if cached_results:
            return cached_results

        # Setup video processing
        video_props = self._setup_video_processing(video_path)

        # Initialize tracking data structures
        tracking_data = self._initialize_tracking_data()

        # Setup video writer if needed
        video_writer = self._setup_video_writer(output_video_path, video_props)

        # Process all frames
        frame_count = self._process_all_frames(
            video_path, video_props, video_writer, tracking_data
        )

        # Generate final results
        self.analysis_results = self._create_analysis_results(
            tracking_data["player_tracks"],
            tracking_data["referee_tracks"],
            tracking_data["ball_tracks"],
            tracking_data["camera_movements"],
            tracking_data["possession_history"],
            video_props,
        )

        # Save results if caching enabled
        self._save_cache_if_enabled()

        print("Video analysis completed successfully")
        return self.analysis_results

    def _process_frame(
        self, frame: np.ndarray, frame_number: int, fps: float
    ) -> Dict[str, Any]:
        """Process a single frame and return all analysis results."""

        # Core detection and tracking
        detections = self.detector.detect(frame)
        tracked_detections = self.tracker.update(detections)
        camera_movement = self.camera_tracker.track_movement(frame)

        # Separate detections by type
        detection_groups = self._group_detections_by_type(tracked_detections)

        # Create player states with all calculations
        player_states = self._create_player_states(
            detection_groups["player"], frame, frame_number, fps
        )

        # Update position tracking for speed calculations
        self._update_position_tracking(player_states)

        # Analyze ball possession
        possession_info = self.possession_analyzer.analyze_possession(
            detection_groups["ball"], player_states
        )

        # Handle team color analysis
        self._handle_team_color_analysis(
            frame, frame_number, player_states, detection_groups["player"]
        )
        team_colors = self.team_analyzer.get_team_colors()

        return {
            "frame_number": frame_number,
            "player_states": player_states,
            "ball_detections": detection_groups["ball"],
            "referee_detections": detection_groups["referee"],
            "camera_movement": camera_movement,
            "possession_info": possession_info,
            "team_colors": team_colors,
        }

    def _group_detections_by_type(
        self, tracked_detections: List[Detection]
    ) -> Dict[str, List[Detection]]:
        """Group detections by object type."""
        return {
            "player": [
                d for d in tracked_detections if d.object_type == ObjectType.PLAYER
            ],
            "ball": [d for d in tracked_detections if d.object_type == ObjectType.BALL],
            "referee": [
                d for d in tracked_detections if d.object_type == ObjectType.REFEREE
            ],
        }

    def _create_player_states(
        self,
        player_detections: List[Detection],
        frame: np.ndarray,
        frame_number: int,
        fps: float,
    ) -> List[PlayerState]:
        """Create player states with position calculations and team assignment."""
        player_states = []

        for detection in player_detections:
            if detection.track_id is None:
                continue

            # Position calculations
            adjusted_position = self.camera_tracker.get_adjusted_position(
                detection.bbox.center, frame_number
            )
            field_position = self.coordinate_transformer.transform_point(
                adjusted_position
            )

            # Speed and distance calculations
            speed, distance = self._calculate_movement_metrics(
                detection.track_id, field_position, fps
            )

            # Team assignment
            team_assignment = self._assign_player_team(frame, detection)

            # Create player state
            player_state = PlayerState(
                track_id=detection.track_id,
                bbox=detection.bbox,
                team=team_assignment,
                position=detection.bbox.center,
                position_adjusted=adjusted_position,
                position_transformed=field_position,
                speed=speed,
                distance=distance,
            )
            player_states.append(player_state)

        return player_states

    def _calculate_movement_metrics(
        self, track_id: int, current_position: Tuple[float, float], fps: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate speed and distance based on previous position."""
        if track_id not in self._previous_positions:
            return None, None

        prev_pos = self._previous_positions[track_id]
        time_diff = 1.0 / fps

        speed = self.coordinate_transformer.calculate_speed(
            prev_pos, current_position, time_diff
        )
        distance = self.coordinate_transformer.calculate_distance(
            prev_pos, current_position
        )

        return speed, distance

    def _assign_player_team(
        self, frame: np.ndarray, detection: Detection
    ) -> TeamAssignment:
        """Assign team to player based on team color analysis."""
        if not self._team_colors_analyzed:
            return TeamAssignment.UNKNOWN

        team_assignment_id = self.team_analyzer.assign_player_team(frame, detection)

        if team_assignment_id == 0:
            return TeamAssignment.TEAM_1
        elif team_assignment_id == 1:
            return TeamAssignment.TEAM_2
        else:
            return TeamAssignment.UNKNOWN

    def _update_position_tracking(self, player_states: List[PlayerState]) -> None:
        """Update position tracking for next frame's speed calculation."""
        for player_state in player_states:
            if (
                player_state.track_id is not None
                and player_state.position_transformed is not None
            ):
                self._previous_positions[player_state.track_id] = (
                    player_state.position_transformed
                )

    def _handle_team_color_analysis(
        self,
        frame: np.ndarray,
        frame_number: int,
        player_states: List[PlayerState],
        player_detections: List[Detection],
    ) -> None:
        """Handle team color analysis when conditions are met."""
        if (
            not self._team_colors_analyzed
            and len(player_states) >= 4
            and self._should_analyze_team_colors_in_frame(frame_number, player_states)
        ):
            self.team_analyzer.analyze_frame_colors(frame, player_detections)
            self._team_colors_analyzed = True

    def _store_frame_results(
        self,
        frame_results: Dict[str, Any],
        frame_number: int,
        player_tracks: Dict[int, List[PlayerState]],
        referee_tracks: Dict[int, List[Dict[str, Any]]],
        ball_tracks: List[Dict[int, Dict[str, Any]]],
        camera_movements: List[List[float]],
        possession_history: List[Any],
    ):
        """Store frame results in tracking dictionaries."""
        self._store_player_tracks(frame_results["player_states"], player_tracks)
        self._store_referee_tracks(frame_results["referee_detections"], referee_tracks)
        self._store_ball_tracks(
            frame_results["ball_detections"], ball_tracks, frame_number
        )

        # Store simple data
        camera_movements.append(frame_results["camera_movement"])
        possession_history.append(frame_results["possession_info"])

    def _store_player_tracks(
        self,
        player_states: List[PlayerState],
        player_tracks: Dict[int, List[PlayerState]],
    ):
        """Store player tracking data."""
        for player_state in player_states:
            if player_state.track_id not in player_tracks:
                player_tracks[player_state.track_id] = []
            player_tracks[player_state.track_id].append(player_state)

    def _store_referee_tracks(
        self,
        referee_detections: List[Detection],
        referee_tracks: Dict[int, List[Dict[str, Any]]],
    ):
        """Store referee tracking data."""
        for referee in referee_detections:
            if referee.track_id is not None:  # Only store if track_id is valid
                if referee.track_id not in referee_tracks:
                    referee_tracks[referee.track_id] = []
                referee_tracks[referee.track_id].append(
                    {"bbox": referee.bbox.as_list(), "confidence": referee.confidence}
                )

    def _store_ball_tracks(
        self,
        ball_detections: List[Detection],
        ball_tracks: List[Dict[int, Dict[str, Any]]],
        frame_number: int,
    ):
        """Store ball tracking data."""
        ball_frame_data = {}
        for ball in ball_detections:
            if ball.track_id is not None:
                ball_frame_data[ball.track_id] = {
                    "bbox": ball.bbox.as_list(),
                    "confidence": ball.confidence,
                }
        if ball_frame_data:
            ball_tracks.append({frame_number: ball_frame_data})

    def _render_frame(
        self, frame: np.ndarray, frame_results: Dict[str, Any]
    ) -> np.ndarray:
        """Render frame with all annotations."""
        return self.renderer.render_frame(
            frame=frame,
            player_states=frame_results["player_states"],
            ball_detections=frame_results["ball_detections"],
            referee_detections=frame_results["referee_detections"],
            team_colors=frame_results["team_colors"],
            possession_info=frame_results["possession_info"],
            camera_movement=frame_results["camera_movement"],
        )

    def _analyze_team_colors(
        self, player_tracks: Dict[int, List[PlayerState]], frame: np.ndarray
    ):
        """Analyze team colors from accumulated player data using the provided frame."""
        # Get recent player states for color analysis
        recent_players = []
        for track_id, states in player_tracks.items():
            if states:  # Get most recent state
                recent_players.append(states[-1])

        if len(recent_players) >= 4:  # Need minimum players for team analysis
            # Create dummy detections for color analysis
            dummy_detections = []
            for player in recent_players:
                detection = Detection(
                    bbox=player.bbox,
                    object_type=ObjectType.PLAYER,
                    track_id=player.track_id,
                    confidence=1.0,
                )
                dummy_detections.append(detection)

            # Analyze colors using the actual frame
            self.team_analyzer.analyze_frame_colors(frame, dummy_detections)

    def _create_analysis_results(
        self,
        player_tracks: Dict[int, List[PlayerState]],
        referee_tracks: Dict[int, List[Dict[str, Any]]],
        ball_tracks: List[Dict[int, Dict[str, Any]]],
        camera_movements: List[List[float]],
        possession_history: List[Any],
        video_props: Dict[str, Any],
    ) -> MatchAnalysis:
        """Create final analysis results."""

        # Calculate team ball control
        # Create frame-by-frame player states for possession analysis
        frame_player_states = []
        max_frames = (
            max(len(states) for states in player_tracks.values())
            if player_tracks
            else 0
        )

        for frame_idx in range(max_frames):
            frame_players = []
            for track_id, states in player_tracks.items():
                if frame_idx < len(states):
                    frame_players.append(states[frame_idx])
            frame_player_states.append(frame_players)

        team_possession_stats = self.possession_analyzer.get_possession_stats(
            frame_player_states
        )

        # Create team ball control array (simplified)
        team_ball_control = np.array(
            [
                team_possession_stats.get("team_1_possession", 0),
                team_possession_stats.get("team_2_possession", 0),
            ]
        )

        return MatchAnalysis(
            player_tracks=player_tracks,
            referee_tracks=referee_tracks,
            ball_tracks=ball_tracks,
            team_ball_control=team_ball_control,
            camera_movement=camera_movements,
            total_frames=len(camera_movements),
            fps=video_props["fps"],
        )

    def _get_previous_positions(self) -> Dict[int, Tuple[float, float]]:
        """Get previous positions for speed calculation."""
        return self._previous_positions

    def get_analysis_results(self) -> Optional[MatchAnalysis]:
        """Get the current analysis results."""
        return self.analysis_results

    def save_results(self, filepath: str) -> None:
        """Save analysis results to file."""
        if self.analysis_results is None:
            raise ValueError("No analysis results to save. Run process_video first.")

        with open(filepath, "wb") as f:
            pickle.dump(self.analysis_results, f)
        print(f"Analysis results saved to: {filepath}")

    def load_results(self, filepath: str):
        """Load analysis results from file."""
        with open(filepath, "rb") as f:
            self.analysis_results = pickle.load(f)

    def _try_load_cache(self) -> Optional[MatchAnalysis]:
        """Try to load cached analysis results."""
        cache_path = os.path.join(
            self.config.processing.output_directory, "analysis_cache.pkl"
        )

        if not (self.config.processing.load_from_cache and os.path.exists(cache_path)):
            return None

        print("Loading cached analysis results...")
        try:
            with open(cache_path, "rb") as f:
                cached_results = pickle.load(f)
            self.analysis_results = cached_results
            print("Cached results loaded successfully")
            return cached_results
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None

    def _setup_video_processing(self, video_path: str) -> Dict[str, Any]:
        """Setup video processing and get video properties."""
        video_props = get_video_properties(video_path)
        print(
            f"Video properties: {video_props}"
        )  # Get keypoints from coordinate transformer
        keypoints = self.coordinate_transformer.get_field_corners()

        if keypoints:
            print(f"Using field corners: {self.get_current_keypoint_strategy()}")
            # Coordinate transformer is already calibrated if corners were set
        elif self.config.transformation.auto_calibrate:
            print(
                "No field corners available - coordinate transformation will use fallback mode"
            )
        else:
            print("No field keypoints available and auto-calibration disabled")

        return video_props

    def _initialize_tracking_data(self) -> Dict[str, Any]:
        """Initialize all tracking data structures."""
        self._team_colors_analyzed = False
        self._previous_positions: Dict[int, Tuple[float, float]] = {}

        return {
            "player_tracks": {},
            "referee_tracks": {},
            "ball_tracks": [],
            "camera_movements": [],
            "possession_history": [],
        }

    def _setup_video_writer(
        self, output_video_path: Optional[str], video_props: Dict[str, Any]
    ) -> Optional[VideoWriter]:
        """Setup video writer if output path is provided."""
        if not output_video_path:
            return None

        return VideoWriter(
            output_video_path,
            fps=video_props["fps"],
            frame_size=(video_props["width"], video_props["height"]),
        )

    def _process_all_frames(
        self,
        video_path: str,
        video_props: Dict[str, Any],
        video_writer: Optional[VideoWriter],
        tracking_data: Dict[str, Any],
    ) -> int:
        """Process all video frames and return total frame count."""
        frame_count = 0

        try:
            with video_writer if video_writer else contextlib.nullcontext():
                for frame in tqdm(
                    read_video_frames(video_path),
                    total=video_props["frame_count"],
                    desc="Processing frames",
                ):
                    # Process single frame
                    frame_results = self._process_frame(
                        frame, frame_count, video_props["fps"]
                    )

                    # Store results
                    self._store_frame_results(
                        frame_results,
                        frame_count,
                        tracking_data["player_tracks"],
                        tracking_data["referee_tracks"],
                        tracking_data["ball_tracks"],
                        tracking_data["camera_movements"],
                        tracking_data["possession_history"],
                    )

                    # Render frame if output requested
                    if video_writer:
                        annotated_frame = self._render_frame(frame, frame_results)
                        video_writer.write_frame(annotated_frame)

                    frame_count += 1

        except Exception as e:
            print(f"Error during video processing: {e}")
            raise

        return frame_count

    def _save_cache_if_enabled(self) -> None:
        """Save analysis results to cache if caching is enabled."""
        if not self.config.processing.save_to_cache:
            return

        try:
            cache_path = os.path.join(
                self.config.processing.output_directory, "analysis_cache.pkl"
            )
            with open(cache_path, "wb") as f:
                pickle.dump(self.analysis_results, f)
            print("Analysis results cached successfully")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def _should_analyze_team_colors(
        self,
        current_frame: int,
        total_frames: int,
        player_tracks: Dict[int, List[PlayerState]],
    ) -> bool:
        """
        Determine if team colors should be analyzed based on tracking stability and video progress.

        Args:
            current_frame: Current frame number
            total_frames: Total frames in video
            player_tracks: Current player tracking data

        Returns:
            True if team colors should be analyzed now
        """
        # Need minimum number of tracked players
        if len(player_tracks) < 4:
            return False

        # Calculate progress through video
        progress = current_frame / total_frames

        # For very short videos (< 30 frames), analyze at 70% through
        if total_frames < 30:
            return progress >= 0.7

        # For short videos (30-100 frames), analyze at 60% through
        elif total_frames < 100:
            return progress >= 0.6

        # For medium videos (100-300 frames), analyze at 40% through
        elif total_frames < 300:
            return progress >= 0.4

        # For long videos, analyze when we have stable tracking (at least 30 frames of data)
        # but not too late (max 30% through video)
        else:
            min_frames_for_stability = 30
            max_progress_for_analysis = 0.3

            return (
                current_frame >= min_frames_for_stability
                and progress <= max_progress_for_analysis
                and self._has_stable_tracking(player_tracks)
            )

    def _has_stable_tracking(self, player_tracks: Dict[int, List[PlayerState]]) -> bool:
        """Check if we have stable player tracking data."""
        # Check that we have players with consistent tracking
        stable_tracks = 0
        for track_id, states in player_tracks.items():
            if len(states) >= 10:  # Player tracked for at least 10 frames
                stable_tracks += 1

        return stable_tracks >= 4  # At least 4 players with stable tracking

    def _should_analyze_team_colors_in_frame(
        self, frame_number: int, player_states: List[PlayerState]
    ) -> bool:
        """
        Simplified check for team color analysis within _process_frame.
        Uses frame-level data instead of accumulated tracks.
        """
        # Basic check: need enough players detected in current frame
        if len(player_states) < 4:
            return False

        # For frame-level analysis, use a simple threshold
        # Analyze after we've seen some frames but not too late
        return (
            frame_number >= 30 and frame_number % 50 == 0
        )  # Every 50 frames starting from frame 30    def set_field_keypoints(self, keypoints: List[List[float]]) -> None:
        """
        Set field keypoints for coordinate transformation.
        
        Args:
            keypoints: List of [x, y] coordinates for field corners
                      Should be in order: [top_left, top_right, bottom_right, bottom_left]
        """
        self.config.transformation.default_keypoints = keypoints
        self.coordinate_transformer.set_field_corners(keypoints)
        print("Field keypoints updated in configuration")

    def get_field_keypoints(self) -> Optional[List[List[float]]]:
        """
        Get current field keypoints.

        Returns:
            Field keypoints if set, None otherwise
        """
        return self.coordinate_transformer.get_field_corners()

    def set_field_keypoints(
        self, keypoints: Optional[List[List[float]]] = None
    ) -> None:
        """
        Set field keypoints for calibration.

        Args:
            keypoints: Optional keypoints to use, if None uses config keypoints
        """
        target_keypoints = keypoints or self.config.transformation.default_keypoints
        if target_keypoints:
            self.config.transformation.default_keypoints = target_keypoints
            self.coordinate_transformer.set_field_corners(target_keypoints)

    def switch_to_auto_detection(self, algorithm: str = "line_detection") -> None:
        """
        Enable automatic keypoint detection (placeholder for future implementation).

        Args:
            algorithm: Detection algorithm to use (reserved for future)
        """
        print(
            f"Auto detection with {algorithm} not yet implemented. Manual keypoints only."
        )

    def switch_to_hybrid_strategy(self, algorithm: str = "line_detection") -> None:
        """
        Enable hybrid mode (placeholder for future implementation).

        Args:
            algorithm: Detection algorithm to use for auto detection (reserved for future)
        """
        print(
            f"Hybrid mode with {algorithm} not yet implemented. Manual keypoints only."
        )
        # Set up manual keypoints if available
        if self.config.transformation.default_keypoints:
            self.coordinate_transformer.set_field_corners(
                self.config.transformation.default_keypoints
            )

    def get_current_keypoint_strategy(self) -> str:
        """Get the name of the current keypoint mode."""
        has_keypoints = self.coordinate_transformer.get_field_corners() is not None

        if has_keypoints:
            return "Field Keypoints Set"
        else:
            return "No Keypoints"


# Context manager import
import contextlib
