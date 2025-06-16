"""
Modern Football Analysis Pipeline
"""


from typing import List, Dict, Optional, Tuple, Any
import pickle
import os.path
import contextlib
import numpy as np
from tqdm import tqdm
import logging

from .config import FootballAIConfig, get_default_config
from .domain.models import (
    Detection,
    FieldEntityState,
    FieldEntityType,
    ObjectType,
    TeamAssignment,
    TeamColor,
    MatchAnalysis,
)
from .detection.yolo_detector import YOLODetector
from .detection.yolo_keypoint_detector import YOLOKeypointDetector
from .tracking.byte_tracker import ByteTracker
from .analysis.team_color_analyzer import KMeansTeamColorAnalyzer
from .assignment.team_assigner import TeamAssigner
from .analysis.ball_possession_analyzer import DistanceBasedBallPossessionAnalyzer
from .motion.camera_motion_tracker import OpticalFlowCameraTracker
from .transformation.coordinate_transformer import PerspectiveCoordinateTransformer
from .rendering.video_renderer import VideoRenderer
from .utils.video_utils import read_video_frames, get_video_properties, VideoWriter

logger = logging.getLogger("football_ai.pipeline")


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
            logger.warning("Configuration warnings:")
            for issue in validation_issues:
                logger.warning(f"  - {issue}")

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
        self.team_assigner = TeamAssigner()
        self.possession_analyzer = DistanceBasedBallPossessionAnalyzer()
        self.camera_tracker = OpticalFlowCameraTracker()
        self.coordinate_transformer = PerspectiveCoordinateTransformer(
            field_corners=self.config.transformation.default_keypoints
        )
        self.renderer = VideoRenderer()

        # State
        self.analysis_results: Optional[MatchAnalysis] = None
        self.is_initialized = False
        self._team_colors_analyzed = (
            False  # Track if we've done initial team color analysis
        )

        if self.config.verbose_logging:
            logger.info("Football AI analysis pipeline initialized successfully!")
            if self.config.debug_mode:
                logger.info(self.config.get_summary())

    def process_video(
        self, video_path: str, output_video_path: Optional[str] = None
    ) -> MatchAnalysis:
        """
        Process a football video and generate complete analysis.

        Main pipeline steps:
        - Check for cached results
        - Setup video processing environment
        - Process each frame sequentially
        - Generate final analysis results
        - Save results and cache
        """
        logger.info(f"Starting analysis of video: {video_path}")

        # Check for cached results first
        cached_results = self._check_cache()
        if cached_results:
            return cached_results

        # Setup video processing environment
        video_props, tracking_data, video_writer = self._setup_processing(
            video_path, output_video_path
        )

        # Process each frame sequentially
        frame_count = self._process_all_frames(
            video_path, video_props, video_writer, tracking_data
        )

        # Generate final analysis results
        self.analysis_results = self._generate_final_results(tracking_data, video_props)

        # Save results and cache
        self._save_results()

        logger.info("Video analysis completed successfully")
        return self.analysis_results

    # ==================== MAIN PIPELINE STEPS ====================

    def _check_cache(self) -> Optional[MatchAnalysis]:
        """
        Check for cached analysis results.

        Returns cached results if available and caching is enabled,
        otherwise returns None to proceed with fresh analysis.
        """
        cache_path = os.path.join(
            self.config.processing.output_directory, "analysis_cache.pkl"
        )
        if not (self.config.processing.load_from_cache and os.path.exists(cache_path)):
            return None

        logger.info("Loading cached analysis results...")
        try:
            with open(cache_path, "rb") as f:
                cached_results = pickle.load(f)
            self.analysis_results = cached_results
            logger.info("Cached results loaded successfully")
            return cached_results
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None

    def _setup_processing(
        self, video_path: str, output_video_path: Optional[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[VideoWriter]]:
        """
        Setup video processing environment.

        Initializes video properties, tracking data structures, and output writer.
        Prepares the pipeline for frame-by-frame processing.
        """
        logger.info("Setting up video processing environment...")

        # Get video properties
        video_props = get_video_properties(video_path)
        logger.info(f"Video properties: {video_props}")

        # Setup field keypoints
        keypoints = self.coordinate_transformer.get_field_corners()
        if keypoints:
            logger.info("Using configured field keypoints")
        else:
            logger.info("No field keypoints configured")

        # Initialize tracking data structures
        self._previous_positions: Dict[int, Tuple[float, float]] = {}
        self._team_assignments_by_track_id = {}
        self._team_colors_analyzed = False

        tracking_data = {
            "field_entity_tracks": {},
            "ball_tracks": [],
            "camera_movements": [],
            "possession_history": [],
        }

        # Setup video writer for output
        video_writer = None
        if output_video_path:
            video_writer = VideoWriter(
                output_video_path,
                fps=video_props["fps"],
                frame_size=(video_props["width"], video_props["height"]),
            )
            logger.info(f"Output video will be saved to: {output_video_path}")

        return video_props, tracking_data, video_writer

    def _process_all_frames(
        self,
        video_path: str,
        video_props: Dict[str, Any],
        video_writer: Optional[VideoWriter],
        tracking_data: Dict[str, Any],
    ) -> int:
        """
        Process each video frame sequentially.

        Main processing loop that handles:
        - Object detection and tracking
        - Team color analysis (one-time)
        - Player team assignment
        - Position calculations
        - Ball possession analysis
        - Frame rendering (optional)
        """
        logger.info("Starting frame-by-frame processing...")
        frame_count = 0

        try:
            with video_writer if video_writer else contextlib.nullcontext():
                for frame in tqdm(
                    read_video_frames(video_path),
                    total=video_props["frame_count"],
                    desc="Processing frames",
                ):
                    # Process single frame through complete pipeline
                    frame_results = self._process_single_frame(
                        frame, frame_count, video_props["fps"]
                    )

                    # Store frame results in tracking data
                    self._store_frame_results_in_tracking_data(
                        frame_results, frame_count, tracking_data
                    )

                    # Render annotated frame if output requested
                    if video_writer:
                        annotated_frame = self._render_annotated_frame(
                            frame, frame_results
                        )
                        video_writer.write_frame(annotated_frame)

                    frame_count += 1

        except Exception as e:
            logger.error(f"Error during video processing: {e}")
            raise

        logger.info(f"Processed {frame_count} frames successfully")
        return frame_count

    def _generate_final_results(
        self, tracking_data: Dict[str, Any], video_props: Dict[str, Any]
    ) -> MatchAnalysis:
        """
        Generate final analysis results.

        Compiles all frame-by-frame data into comprehensive match analysis,
        including possession statistics and movement analytics.
        """
        logger.info("Generating final analysis results...")

        # Extract components from tracking data
        field_entity_tracks = tracking_data["field_entity_tracks"]
        ball_tracks = tracking_data["ball_tracks"]
        camera_movements = tracking_data["camera_movements"]
        possession_history = tracking_data["possession_history"]

        # Calculate team possession statistics
        team_possession_stats = self._calculate_team_possession_statistics(
            field_entity_tracks
        )

        # Create team ball control array
        team_ball_control = np.array(
            [
                team_possession_stats.get("team_1_possession", 0),
                team_possession_stats.get("team_2_possession", 0),
            ]
        )

        # Generate comprehensive match analysis
        analysis = MatchAnalysis(
            field_entity_tracks=field_entity_tracks,
            ball_tracks=ball_tracks,
            team_ball_control=team_ball_control,
            camera_movement=camera_movements,
            total_frames=len(camera_movements),
            fps=video_props["fps"],
        )

        logger.info(
            f"Analysis complete: {len(field_entity_tracks)} entity tracks, "
            f"{len(ball_tracks)} ball tracks, {analysis.total_frames} frames"
        )

        return analysis

    def _save_results(self) -> None:
        """
        Save analysis results and cache.

        Saves the complete analysis to cache file if caching is enabled.
        """
        if not self.config.processing.save_to_cache:
            logger.info("Caching disabled, skipping cache save")
            return

        try:
            cache_path = os.path.join(
                self.config.processing.output_directory, "analysis_cache.pkl"
            )
            with open(cache_path, "wb") as f:
                pickle.dump(self.analysis_results, f)
            logger.info(f"Analysis results cached successfully to: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    # ==================== FRAME PROCESSING METHODS ====================

    def _process_single_frame(
        self, frame: np.ndarray, frame_number: int, fps: float
    ) -> Dict[str, Any]:
        """
        Process a single frame through the complete analysis pipeline.

        Steps:
        - Detect and track objects
        - Analyze team colors (one-time setup)
        - Assign players to teams
        - Calculate positions and movement
        - Analyze ball possession
        """
        # Core detection and tracking
        detections = self.detector.detect(frame)
        tracked_detections = self.tracker.update(detections)
        camera_movement = self.camera_tracker.track_movement(frame)

        # Group detections by type
        detection_groups = self._group_detections_by_type(tracked_detections)

    
        field_players = detection_groups["player"] + detection_groups["goalkeeper"]
        players = detection_groups["player"]
        
        if not self._team_colors_analyzed:
            # Initial team color analysis and batch assignment
            if len(players) >= 4:
                logger.info(
                    f"Analyzing team colors with {len(players)} players at frame {frame_number}"
                )
                team_features = self.team_analyzer.analyze_team_features(
                    frame, players
                )
                if team_features and len(team_features) >= 2:
                    self.team_assigner.set_team_features(team_features)
                    assignments = self.team_assigner.assign_players_batch(
                        frame, players
                    )
                    self._team_colors_analyzed = True
                    logger.info(
                        f"Team colors analyzed and assigned {len(assignments)} players"
                    )
                    stats = self.team_assigner.get_assignment_stats()
                    logger.info(f"Assignment stats: {stats}")
                else:
                    logger.warning("Failed to identify distinct team colors")
        else:
            # Assign new players to established teams
            new_players_assigned = 0
            for detection in players:
                if (
                    detection.track_id is not None
                    and self.team_assigner.get_player_team_assignment(
                        detection.track_id
                    )
                    is None
                ):
                    assignment = self.team_assigner.assign_player_team(frame, detection)
                    if assignment is not None:
                        new_players_assigned += 1
            if new_players_assigned > 0:
                logger.info(f"Assigned {new_players_assigned} new players to teams")

        # Create entity states with position calculations
        player_entities = self._create_player_states(
            field_players, frame, frame_number, fps
        )
        referee_entities = self._create_referee_states(
            detection_groups["referee"], frame, frame_number, fps
        )

        # Update position tracking for movement calculations
        all_field_entities = player_entities + referee_entities
        self._update_position_tracking(all_field_entities)

        # Analyze ball possession
        possession_info = self.possession_analyzer.analyze_possession(
            detection_groups["ball"], player_entities
        )

        return {
            "frame_number": frame_number,
            "field_entities": all_field_entities,
            "player_entities": player_entities,
            "ball_detections": detection_groups["ball"],
            "camera_movement": camera_movement,
            "possession_info": possession_info,
            "team_colors": (
                self._convert_team_features_to_colors(self.team_assigner._team_features)
                if self.team_assigner.has_team_features()
                else None
            ),
        }

    def _store_frame_results_in_tracking_data(
        self,
        frame_results: Dict[str, Any],
        frame_number: int,
        tracking_data: Dict[str, Any],
    ) -> None:
        """
        Store frame processing results in tracking data structures.

        Organizes results by entity tracks, ball tracks, camera movement,
        and possession history for final analysis generation.
        """
        # Store field entity tracks
        self._store_field_entity_tracks(
            frame_results["field_entities"], tracking_data["field_entity_tracks"]
        )

        # Store ball tracking data
        self._store_ball_tracks(
            frame_results["ball_detections"], tracking_data["ball_tracks"], frame_number
        )

        # Store camera movement and possession data
        tracking_data["camera_movements"].append(frame_results["camera_movement"])
        tracking_data["possession_history"].append(frame_results["possession_info"])

    def _store_field_entity_tracks(
        self,
        field_entities: List[FieldEntityState],
        field_entity_tracks: Dict[int, List[FieldEntityState]],
    ) -> None:
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
    ) -> None:
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

    def _render_annotated_frame(
        self, frame: np.ndarray, frame_results: Dict[str, Any]
    ) -> np.ndarray:
        """
        Render frame with all analysis annotations.

        Creates visual output showing:
        - Player bounding boxes with team colors
        - Ball tracking and possession indicators
        - Movement trails and statistics
        """
        # Convert field entities back to referee detections for renderer compatibility
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
            player_states=frame_results["player_entities"],
            ball_detections=frame_results["ball_detections"],
            referee_detections=referee_detections,
            team_colors=frame_results["team_colors"],
            possession_info=frame_results["possession_info"],
            camera_movement=frame_results["camera_movement"],
        )

    def _calculate_team_possession_statistics(
        self, field_entity_tracks: Dict[int, List[FieldEntityState]]
    ) -> Dict[str, float]:
        """
        Calculate team possession statistics from field entity tracks.

        Analyzes player movement and ball possession data to generate
        comprehensive team control percentages.
        """
        # Extract player tracks only (no referees for possession analysis)
        player_tracks = {}
        for track_id, entity_track in field_entity_tracks.items():
            player_entities = [entity for entity in entity_track if entity.is_player]
            if player_entities:
                player_tracks[track_id] = player_entities

        # Create frame-by-frame player states for possession analysis
        max_frames = (
            max(len(states) for states in player_tracks.values())
            if player_tracks
            else 0
        )

        frame_player_states = []
        for frame_idx in range(max_frames):
            frame_players = []
            for track_id, states in player_tracks.items():
                if frame_idx < len(states):
                    frame_players.append(states[frame_idx])
            frame_player_states.append(frame_players)

        # Generate possession statistics
        return self.possession_analyzer.get_possession_stats(frame_player_states)

    # ==================== OBJECT DETECTION AND TRACKING METHODS ====================

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
        """Assign team to player using team assigner."""
        if detection.track_id is not None:
            # Get assignment from team assigner
            assignment = self.team_assigner.get_player_team_assignment(
                detection.track_id
            )
            if assignment is not None:
                return assignment

            # If not assigned yet, try to assign now
            assignment = self.team_assigner.assign_player_team(frame, detection)
            if assignment is not None:
                return assignment

        # Fallback to unknown if assignment fails
        return TeamAssignment.UNKNOWN

    def _update_position_tracking(self, field_entities: List[FieldEntityState]) -> None:
        """Update position tracking for next frame's speed calculation."""
        for entity in field_entities:
            if entity.track_id is not None and entity.position_transformed is not None:
                self._previous_positions[entity.track_id] = entity.position_transformed

    # ==================== ENTITY STATE CREATION METHODS ====================

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

    # ==================== UTILITY AND HELPER METHODS ====================

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
        logger.info(f"Analysis results saved to: {filepath}")

    def load_results(self, filepath: str):
        """Load analysis results from file."""
        with open(filepath, "rb") as f:
            self.analysis_results = pickle.load(f)

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
            "team_features": (
                self.team_assigner._team_features
                if hasattr(self, "team_assigner")
                and self.team_assigner.has_team_features()
                else None
            ),
            "assignments": (
                self.team_assigner.get_assignment_stats()
                if hasattr(self, "team_assigner")
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

    def _convert_team_features_to_colors(
        self, team_features: Optional[Dict[int, Any]]
    ) -> Optional[Dict[int, TeamColor]]:
        """
        Convert TeamFeatures objects to TeamColor objects for renderer compatibility.

        Args:
            team_features: Dictionary of team features from the team assigner

        Returns:
            Dictionary of TeamColor objects for the renderer, or None if no features
        """
        if not team_features:
            return None

        team_colors = {}
        for team_id, features in team_features.items():
            if hasattr(features, "get_feature") and features.get_feature("color"):
                color = features.get_feature("color")
                team_colors[team_id] = TeamColor(
                    id=team_id, primary_color=color, name=f"Team_{team_id}"
                )

        return team_colors if team_colors else None
