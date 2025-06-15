"""
Modern Football Analysis Pipeline
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import pickle
import os.path
import contextlib
from tqdm import tqdm

from .config import FootballAIConfig, get_default_config
from .domain.models import (
    Detection,
    FieldEntityState,
    FieldEntityType,
    ObjectType,
    TeamAssignment,
    MatchAnalysis,
)
from .detection.yolo_detector import YOLODetector
from .detection.yolo_keypoint_detector import YOLOKeypointDetector
from .tracking.byte_tracker import ByteTracker
from .analysis.team_color_analyzer import KMeansTeamColorAnalyzer
from .analysis.ball_possession_analyzer import DistanceBasedBallPossessionAnalyzer
from .motion.camera_motion_tracker import OpticalFlowCameraTracker
from .transformation.coordinate_transformer import PerspectiveCoordinateTransformer
from .rendering.video_renderer import VideoRenderer
from .utils.video_utils import read_video_frames, get_video_properties, VideoWriter


class FootballAnalysisPipeline:
    """
    Football analysis pipeline that integrates all components for video analysis.

    Usage:
        pipeline = FootballAnalysisPipeline()
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
        """Initialize the football analysis pipeline."""
        # Initialize configuration
        self.config = config if config is not None else get_default_config()

        # Override config with explicit parameters
        if model_path:
            self.config.model.player_model_path = model_path
        if output_dir:
            self.config.processing.output_directory = output_dir
        if save_cache is not None:
            self.config.processing.save_to_cache = save_cache
        if load_cache is not None:
            self.config.processing.load_from_cache = load_cache

        # Validate and create output directory
        validation_issues = self.config.validate()
        if validation_issues:
            print("Configuration warnings:")
            for issue in validation_issues:
                print(f"  - {issue}")

        os.makedirs(self.config.processing.output_directory, exist_ok=True)

        # Initialize components
        self.detector = YOLODetector(
            self.config.model.player_model_path, self.config.model.confidence_threshold
        )

        # Initialize keypoint detector if enabled
        self.keypoint_detector = None
        if self.config.model.enable_keypoint_detection:
            self.keypoint_detector = YOLOKeypointDetector(
                self.config.model.field_model_path,
                self.config.model.keypoint_confidence_threshold,
            )

        self.tracker = ByteTracker()
        self.team_analyzer = KMeansTeamColorAnalyzer()
        self.possession_analyzer = DistanceBasedBallPossessionAnalyzer()
        self.camera_tracker = OpticalFlowCameraTracker()
        self.coordinate_transformer = PerspectiveCoordinateTransformer(
            field_corners=self.config.transformation.default_keypoints
        )
        self.renderer = VideoRenderer()

        # State
        self.analysis_results: Optional[MatchAnalysis] = None
        self.is_initialized = False
        self._team_assignments_by_track_id: Dict[int, TeamAssignment] = (
            {}
        )  # Cache team assignments by track_id
        self._team_colors_analyzed = (
            False  # Track if we've done initial team color analysis
        )

        if self.config.verbose_logging:
            print("Football AI analysis pipeline initialized successfully!")
            if self.config.debug_mode:
                print(self.config.get_summary())

    def process_video(
        self, video_path: str, output_video_path: Optional[str] = None
    ) -> MatchAnalysis:
        """Process a football video and generate complete analysis."""
        print(f"Starting analysis of video: {video_path}")

        # Try to load cached results first
        cached_results = self._try_load_cache()
        if cached_results:
            return cached_results

        # Setup and process
        video_props = self._setup_video_processing(video_path)
        tracking_data = self._initialize_tracking_data()
        video_writer = self._setup_video_writer(output_video_path, video_props)

        frame_count = self._process_all_frames(
            video_path, video_props, video_writer, tracking_data
        )

        # Generate results
        self.analysis_results = self._create_analysis_results(
            tracking_data["field_entity_tracks"],
            tracking_data["ball_tracks"],
            tracking_data["camera_movements"],
            tracking_data["possession_history"],
            video_props,
        )

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

        # Group detections by type
        detection_groups = self._group_detections_by_type(tracked_detections)

        # Get all field players (players + goalkeepers) for team color analysis
        field_players = detection_groups["player"] + detection_groups["goalkeeper"]
        self._handle_team_color_analysis(frame, frame_number, field_players)

        # Create field entity states for all entities
        player_entities = self._create_player_states(
            field_players, frame, frame_number, fps
        )
        referee_entities = self._create_referee_states(
            detection_groups["referee"], frame, frame_number, fps
        )

        # Combine all field entities and update position tracking
        all_field_entities = player_entities + referee_entities
        self._update_position_tracking(all_field_entities)

        # Analyze ball possession with player entities only (referees don't play with ball)
        possession_info = self.possession_analyzer.analyze_possession(
            detection_groups["ball"], player_entities
        )

        return {
            "frame_number": frame_number,
            "field_entities": all_field_entities,  # All field entities (players, goalkeepers, referees)
            "player_entities": player_entities,  # For backward compatibility
            "ball_detections": detection_groups["ball"],
            "camera_movement": camera_movement,
            "possession_info": possession_info,
            "team_colors": self.team_analyzer.get_team_colors(),
        }

    def _group_detections_by_type(
        self, tracked_detections: List[Detection]
    ) -> Dict[str, List[Detection]]:
        """Group detections by object type."""
        return {
            "player": [
                d for d in tracked_detections if d.object_type == ObjectType.PLAYER
            ],
            "goalkeeper": [
                d for d in tracked_detections if d.object_type == ObjectType.GOALKEEPER
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
    ) -> List[FieldEntityState]:
        """Create field entity states with position calculations and team assignment."""
        field_entities = []
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
            speed, distance = self._calculate_movement_metrics(
                detection.track_id, field_position, fps
            )
            team_assignment = self._assign_player_team(frame, detection)

            # Determine entity type based on object type
            if detection.object_type == ObjectType.PLAYER:
                entity_type = FieldEntityType.PLAYER
            elif detection.object_type == ObjectType.GOALKEEPER:
                entity_type = FieldEntityType.GOALKEEPER
            else:
                # This shouldn't happen for player detections, but handle it gracefully
                entity_type = FieldEntityType.PLAYER

            field_entities.append(
                FieldEntityState(
                    track_id=detection.track_id,
                    bbox=detection.bbox,
                    entity_type=entity_type,
                    team=team_assignment,
                    position=detection.bbox.center,
                    position_adjusted=adjusted_position,
                    position_transformed=field_position,
                    speed=speed,
                    distance=distance,
                    team_assignment_confidence="confirmed",
                )
            )
        return field_entities

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
        """Assign team to player using cached assignments or analyzer fallback."""
        if (
            detection.track_id is not None
            and detection.track_id in self._team_assignments_by_track_id
        ):
            # Use cached assignment - much faster!
            return self._team_assignments_by_track_id[detection.track_id]

        # Fallback to analyzer if not in cache (shouldn't happen often after initial analysis)
        team_id = self.team_analyzer.assign_player_team(frame, detection)
        team_assignment = (
            TeamAssignment.TEAM_1
            if team_id == 0
            else TeamAssignment.TEAM_2 if team_id == 1 else TeamAssignment.UNKNOWN
        )

        # Cache the result for future frames
        if detection.track_id is not None:
            self._team_assignments_by_track_id[detection.track_id] = team_assignment

        return team_assignment

    def _update_position_tracking(self, field_entities: List[FieldEntityState]) -> None:
        """Update position tracking for next frame's speed calculation."""
        for entity in field_entities:
            if entity.track_id is not None and entity.position_transformed is not None:
                self._previous_positions[entity.track_id] = entity.position_transformed

    def _handle_team_color_analysis(
        self, frame: np.ndarray, frame_number: int, player_detections: List[Detection]
    ) -> None:
        """Handle team color analysis - analyze once and cache by track ID."""
        # Only analyze if we haven't done it yet and have enough players
        if not self._team_colors_analyzed and len(player_detections) >= 4:
            # Analyze team colors once
            team_colors = self.team_analyzer.analyze_frame_colors(
                frame, player_detections
            )

            if team_colors and len(team_colors) >= 2:
                # Cache team assignments for all current players
                for detection in player_detections:
                    if detection.track_id is not None:
                        team_id = self.team_analyzer.assign_player_team(
                            frame, detection
                        )
                        if team_id == 0:
                            self._team_assignments_by_track_id[detection.track_id] = (
                                TeamAssignment.TEAM_1
                            )
                        elif team_id == 1:
                            self._team_assignments_by_track_id[detection.track_id] = (
                                TeamAssignment.TEAM_2
                            )
                        else:
                            self._team_assignments_by_track_id[detection.track_id] = (
                                TeamAssignment.UNKNOWN
                            )

                self._team_colors_analyzed = True
                print(
                    f"Team colors analyzed and cached for {len(self._team_assignments_by_track_id)} players at frame {frame_number}"
                )

        # Handle new players that weren't in the initial analysis
        elif self._team_colors_analyzed:
            for detection in player_detections:
                if (
                    detection.track_id is not None
                    and detection.track_id not in self._team_assignments_by_track_id
                ):
                    # Assign team for new player using already analyzed team colors
                    team_id = self.team_analyzer.assign_player_team(frame, detection)
                    if team_id == 0:
                        self._team_assignments_by_track_id[detection.track_id] = (
                            TeamAssignment.TEAM_1
                        )
                    elif team_id == 1:
                        self._team_assignments_by_track_id[detection.track_id] = (
                            TeamAssignment.TEAM_2
                        )
                    else:
                        self._team_assignments_by_track_id[detection.track_id] = (
                            TeamAssignment.UNKNOWN
                        )

    def _store_frame_results(
        self,
        frame_results: Dict[str, Any],
        frame_number: int,
        field_entity_tracks: Dict[int, List[FieldEntityState]],
        ball_tracks: List[Dict[int, Dict[str, Any]]],
        camera_movements: List[List[float]],
        possession_history: List[Any],
    ):
        """Store frame results in tracking dictionaries."""
        self._store_field_entity_tracks(
            frame_results["field_entities"], field_entity_tracks
        )
        self._store_ball_tracks(
            frame_results["ball_detections"], ball_tracks, frame_number
        )
        camera_movements.append(frame_results["camera_movement"])
        possession_history.append(frame_results["possession_info"])

    def _store_field_entity_tracks(
        self,
        field_entities: List[FieldEntityState],
        field_entity_tracks: Dict[int, List[FieldEntityState]],
    ):
        """Store field entity tracking data for all entities (players, goalkeepers, referees)."""
        for entity in field_entities:
            if entity.track_id not in field_entity_tracks:
                field_entity_tracks[entity.track_id] = []
            field_entity_tracks[entity.track_id].append(entity)

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
        # Create legacy referee detections for renderer compatibility
        referee_detections = []
        for entity in frame_results["field_entities"]:
            if entity.is_referee:
                referee_detections.append(
                    Detection(
                        bbox=entity.bbox,
                        object_type=ObjectType.REFEREE,
                        track_id=entity.track_id,
                        confidence=1.0,
                    )
                )

        return self.renderer.render_frame(
            frame=frame,
            player_states=frame_results["player_entities"],  # Player entities only
            ball_detections=frame_results["ball_detections"],
            referee_detections=referee_detections,  # Convert back for renderer
            team_colors=frame_results["team_colors"],
            possession_info=frame_results["possession_info"],
            camera_movement=frame_results["camera_movement"],
        )

    def _create_analysis_results(
        self,
        field_entity_tracks: Dict[int, List[FieldEntityState]],
        ball_tracks: List[Dict[int, Dict[str, Any]]],
        camera_movements: List[List[float]],
        possession_history: List[Any],
        video_props: Dict[str, Any],
    ) -> MatchAnalysis:
        """Create final analysis results."""

        # Calculate team ball control from player entities only
        # Create frame-by-frame player states for possession analysis
        frame_player_states = []

        # Extract player tracks from field entity tracks
        player_tracks = {}
        for track_id, entity_track in field_entity_tracks.items():
            player_entities = [entity for entity in entity_track if entity.is_player]
            if player_entities:
                player_tracks[track_id] = player_entities

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
            field_entity_tracks=field_entity_tracks,
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
        print(f"Video properties: {video_props}")

        keypoints = self.coordinate_transformer.get_field_corners()
        if keypoints:
            print("Using configured field keypoints")
        else:
            print("No field keypoints configured")

        return video_props

    def _initialize_tracking_data(self) -> Dict[str, Any]:
        """Initialize all tracking data structures."""
        self._previous_positions: Dict[int, Tuple[float, float]] = {}
        self._team_assignments_by_track_id = {}  # Reset team assignment cache
        self._team_colors_analyzed = False  # Reset team color analysis flag
        return {
            "field_entity_tracks": {},
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
                        tracking_data["field_entity_tracks"],
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

    def get_field_keypoints(self) -> Optional[List[List[float]]]:
        """Get current field keypoints."""
        return self.coordinate_transformer.get_field_corners()

    def set_field_keypoints(
        self, keypoints: Optional[List[List[float]]] = None
    ) -> None:
        """Set field keypoints for coordinate transformation."""
        target_keypoints = keypoints or self.config.transformation.default_keypoints
        if target_keypoints:
            self.config.transformation.default_keypoints = target_keypoints
            self.coordinate_transformer.set_field_corners(target_keypoints)

    def get_team_color_analysis_status(self) -> Dict[str, Any]:
        """Get the current status of team color analysis."""
        return {
            "team_colors": (
                self.team_analyzer.get_team_colors()
                if hasattr(self, "team_analyzer")
                else None
            ),
        }

    def _create_referee_states(
        self,
        referee_detections: List[Detection],
        frame: np.ndarray,
        frame_number: int,
        fps: float,
    ) -> List[FieldEntityState]:
        """Create referee field entity states with basic tracking information."""
        referee_entities = []
        for detection in referee_detections:
            if detection.track_id is None:
                continue

            # Basic position calculations (referees don't need team assignment or advanced metrics)
            adjusted_position = self.camera_tracker.get_adjusted_position(
                detection.bbox.center, frame_number
            )
            field_position = self.coordinate_transformer.transform_point(
                adjusted_position
            )
            speed, distance = self._calculate_movement_metrics(
                detection.track_id, field_position, fps
            )

            referee_entities.append(
                FieldEntityState(
                    track_id=detection.track_id,
                    bbox=detection.bbox,
                    entity_type=FieldEntityType.REFEREE,
                    team=None,  # Referees have no team assignment
                    position=detection.bbox.center,
                    position_adjusted=adjusted_position,
                    position_transformed=field_position,
                    speed=speed,
                    distance=distance,
                    team_assignment_confidence="confirmed",  # Not applicable but required
                )
            )
        return referee_entities
