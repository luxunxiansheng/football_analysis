"""
Modern Football Analysis Pipeline

This module provides the main pipeline that integrates all components
to replicate the functionality of the original notebook workflow.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import pickle
import os
from tqdm import tqdm

from .domain.models import (
    Detection,
    PlayerState,
    ObjectType,
    TeamAssignment,
    MatchAnalysis,
    BoundingBox,
)
from .detection.yolo_detector import ModernYOLODetector
from .tracking.byte_tracker import ModernByteTracker
from .analysis.team_color_analyzer import ModernTeamColorAnalyzer
from .analysis.ball_possession_analyzer import ModernBallPossessionAnalyzer
from .motion.camera_motion_tracker import ModernCameraMotionTracker
from .transformation.coordinate_transformer import ModernCoordinateTransformer
from .rendering.video_renderer import ModernVideoRenderer
from .utils.video_utils import read_video_frames, get_video_properties, VideoWriter


class FootballAnalysisPipeline:
    """
    Modern football analysis pipeline that replicates the original
    notebook functionality with clean architecture.
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str = "output",
        save_cache: bool = True,
        load_cache: bool = True,
    ):
        """
        Initialize the football analysis pipeline.

        Args:
            model_path: Path to YOLO model file
            output_dir: Directory for output files
            save_cache: Whether to save processing cache
            load_cache: Whether to load from cache if available
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.save_cache = save_cache
        self.load_cache = load_cache

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        self.detector = ModernYOLODetector(model_path)
        self.tracker = ModernByteTracker()
        self.team_analyzer = ModernTeamColorAnalyzer()
        self.possession_analyzer = ModernBallPossessionAnalyzer()
        self.camera_tracker = ModernCameraMotionTracker()
        self.coordinate_transformer = ModernCoordinateTransformer()
        self.renderer = ModernVideoRenderer()

        # Analysis results
        self.analysis_results: Optional[MatchAnalysis] = None

        # Processing state
        self.is_initialized = False

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        field_keypoints: Optional[List[List[float]]] = None,
    ) -> MatchAnalysis:
        """
        Process a football video and generate complete analysis.

        Args:
            video_path: Path to input video
            output_video_path: Path for annotated output video
            field_keypoints: Field corner coordinates for transformation

        Returns:
            Complete match analysis results
        """
        print(f"Starting analysis of video: {video_path}")

        # Check for cached results
        cache_path = os.path.join(self.output_dir, "analysis_cache.pkl")
        if self.load_cache and os.path.exists(cache_path):
            print("Loading cached analysis results...")
            try:
                with open(cache_path, "rb") as f:
                    self.analysis_results = pickle.load(f)
                print("Cached results loaded successfully")
                return self.analysis_results
            except Exception as e:
                print(f"Failed to load cache: {e}")

        # Get video properties
        video_props = get_video_properties(video_path)
        print(f"Video properties: {video_props}")

        # Initialize coordinate transformer if keypoints provided
        if field_keypoints:
            self.coordinate_transformer.calibrate(field_keypoints)

        # Process video frames
        player_tracks = {}
        referee_tracks = {}
        ball_tracks = []
        camera_movements = []
        possession_history = []

        frame_count = 0

        # Setup video writer if output path provided
        video_writer = None
        if output_video_path:
            video_writer = VideoWriter(
                output_video_path,
                fps=video_props["fps"],
                frame_size=(video_props["width"], video_props["height"]),
            )

        try:
            with video_writer if video_writer else contextlib.nullcontext():
                # Process frames
                for frame in tqdm(
                    read_video_frames(video_path),
                    total=video_props["frame_count"],
                    desc="Processing frames",
                ):

                    frame_results = self._process_frame(
                        frame, frame_count, video_props["fps"]
                    )

                    # Store results
                    self._store_frame_results(
                        frame_results,
                        frame_count,
                        player_tracks,
                        referee_tracks,
                        ball_tracks,
                        camera_movements,
                        possession_history,
                    )

                    # Render frame if output requested
                    if video_writer:
                        annotated_frame = self._render_frame(frame, frame_results)
                        video_writer.write_frame(annotated_frame)

                    frame_count += 1

                    # Process first few frames for team color analysis
                    if frame_count == 50:  # Analyze team colors after 50 frames
                        self._analyze_team_colors(player_tracks)

        except Exception as e:
            print(f"Error during video processing: {e}")
            raise

        # Create final analysis results
        self.analysis_results = self._create_analysis_results(
            player_tracks,
            referee_tracks,
            ball_tracks,
            camera_movements,
            possession_history,
            video_props,
        )

        # Save cache
        if self.save_cache:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.analysis_results, f)
                print("Analysis results cached successfully")
            except Exception as e:
                print(f"Failed to save cache: {e}")

        print("Video analysis completed successfully")
        return self.analysis_results

    def _process_frame(
        self, frame: np.ndarray, frame_number: int, fps: float
    ) -> Dict[str, Any]:
        """Process a single frame and return all analysis results."""

        # 1. Object detection
        detections = self.detector.detect(frame)

        # 2. Object tracking
        tracked_detections = self.tracker.update(detections)

        # 3. Camera movement tracking
        camera_movement = self.camera_tracker.track_movement(frame)

        # 4. Separate detections by type
        player_detections = [
            d for d in tracked_detections if d.object_type == ObjectType.PLAYER
        ]
        ball_detections = [
            d for d in tracked_detections if d.object_type == ObjectType.BALL
        ]
        referee_detections = [
            d for d in tracked_detections if d.object_type == ObjectType.REFEREE
        ]

        # 5. Create player states
        player_states = []
        for detection in player_detections:
            # Adjust position for camera movement
            adjusted_position = self.camera_tracker.get_adjusted_position(
                detection.bbox.center, frame_number
            )

            # Transform to field coordinates
            field_position = self.coordinate_transformer.transform_point(
                adjusted_position
            )

            # Calculate speed if we have previous position
            speed = None
            distance = None
            if detection.track_id in self._get_previous_positions():
                prev_pos = self._get_previous_positions()[detection.track_id]
                time_diff = 1.0 / fps  # Time between frames
                speed = self.coordinate_transformer.calculate_speed(
                    prev_pos, field_position, time_diff
                )
                distance = self.coordinate_transformer.calculate_distance(
                    prev_pos, field_position
                )

            # Assign team
            team_assignment_id = self.team_analyzer.assign_player_team(frame, detection)
            team_assignment = None
            if team_assignment_id is not None:
                if team_assignment_id == 0:
                    team_assignment = TeamAssignment.TEAM_1
                elif team_assignment_id == 1:
                    team_assignment = TeamAssignment.TEAM_2
                else:
                    team_assignment = TeamAssignment.UNKNOWN

            # Only create player state if we have a valid track ID
            if detection.track_id is not None:
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

        # 6. Ball possession analysis
        possession_info = self.possession_analyzer.analyze_possession(
            ball_detections, player_states
        )

        # 7. Get team colors
        team_colors = self.team_analyzer.get_team_colors()

        return {
            "frame_number": frame_number,
            "player_states": player_states,
            "ball_detections": ball_detections,
            "referee_detections": referee_detections,
            "camera_movement": camera_movement,
            "possession_info": possession_info,
            "team_colors": team_colors,
        }

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

        # Store player tracks
        for player_state in frame_results["player_states"]:
            if player_state.track_id not in player_tracks:
                player_tracks[player_state.track_id] = []
            player_tracks[player_state.track_id].append(player_state)

        # Store referee tracks
        for referee in frame_results["referee_detections"]:
            if referee.track_id not in referee_tracks:
                referee_tracks[referee.track_id] = []
            referee_tracks[referee.track_id].append(
                {"bbox": referee.bbox.as_list(), "confidence": referee.confidence}
            )

        # Store ball tracks
        ball_frame_data = {}
        for ball in frame_results["ball_detections"]:
            if ball.track_id is not None:
                ball_frame_data[ball.track_id] = {
                    "bbox": ball.bbox.as_list(),
                    "confidence": ball.confidence,
                }
        if ball_frame_data:
            ball_tracks.append({frame_number: ball_frame_data})

        # Store camera movement
        camera_movements.append(frame_results["camera_movement"])

        # Store possession info
        possession_history.append(frame_results["possession_info"])

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

    def _analyze_team_colors(self, player_tracks: Dict[int, List[PlayerState]]):
        """Analyze team colors from accumulated player data."""
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

            # Analyze colors (would need a representative frame)
            # This is simplified - in practice you'd use the actual frame
            # self.team_analyzer.analyze_frame_colors(frame, dummy_detections)

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
        # This would be implemented to track previous frame positions
        # For now, return empty dict
        return {}

    def get_analysis_results(self) -> Optional[MatchAnalysis]:
        """Get the current analysis results."""
        return self.analysis_results

    def save_results(self, filepath: str):
        """Save analysis results to file."""
        if self.analysis_results is None:
            raise ValueError("No analysis results to save")

        with open(filepath, "wb") as f:
            pickle.dump(self.analysis_results, f)

    def load_results(self, filepath: str):
        """Load analysis results from file."""
        with open(filepath, "rb") as f:
            self.analysis_results = pickle.load(f)


# Context manager import
import contextlib
