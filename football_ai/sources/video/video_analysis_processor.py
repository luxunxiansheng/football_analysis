"""
Video Analysis Processor - Main interface for video analysis in the game-centric architecture.

This processor coordinates all video analysis and integrates results into Game objects.
"""

from typing import Optional, Dict, Any
from pathlib import Path

from .video_loader import VideoLoader
from .video_pipeline import VideoPipeline
from ...core_models.video import Video
from ...core_models.game import Game, AnalysisSource
from ...utilities import setup_logger


class VideoAnalysisProcessor:
    """
    Main processor for video analysis within the game-centric architecture.

    This class coordinates video loading, analysis, and integration of results
    into Game objects. It serves as the bridge between the old video-centric
    pipeline and the new game-centric approach.
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
        # Optional advanced parameters
        enable_team_classification: bool = True,
        enable_ball_tracking: bool = True,
        team_model_path: Optional[str] = None,
    ):
        """
        Initialize VideoAnalysisProcessor with explicit parameters.

        Args:
            model_path: Path to YOLO detection model
            confidence_threshold: Detection confidence threshold (0.0-1.0)
            iou_threshold: IoU threshold for non-maximum suppression
            device: Device to run models on ("cuda", "cpu", or "mps")
            track_threshold: Tracking confidence threshold
            track_buffer: Number of frames to keep lost tracks
            max_detections: Maximum detections per frame
            log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
            enable_team_classification: Whether to classify team colors
            enable_ball_tracking: Whether to track the ball
            team_model_path: Path to team classification model (optional)
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
        self.team_model_path = team_model_path or "models/embed/siglip-base-patch16-224"

        # Setup logging
        self.logger = setup_logger("VideoAnalysisProcessor", log_level)

        # Initialize components with explicit parameters
        self.loader = VideoLoader(log_level=log_level)
        self.pipeline = VideoPipeline(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device,
            track_threshold=track_threshold,
            track_buffer=track_buffer,
            max_detections=max_detections,
            enable_team_classification=enable_team_classification,
            enable_ball_tracking=enable_ball_tracking,
            enable_video_rendering=False,  # Video rendering processors are added dynamically when output_path is provided
            team_model_path=self.team_model_path,
            log_level=log_level,
        )

    def analyze_video_for_game(
        self,
        game: Game,
        video_path: str,
        max_frames: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> Game:
        """
        Analyze a video and integrate results into a Game object.

        Args:
            game: Game object to integrate analysis into
            video_path: Path to video file
            max_frames: Optional limit on frames to process
            output_path: Optional path for output video

        Returns:
            Updated Game object with video analysis integrated
        """
        self.logger.info(f"Starting video analysis for game {game.game_id}")

        try:
            # Load video
            video = self.loader.load_video(video_path, max_frames)

            # Run video analysis pipeline
            analyzed_video = self.pipeline.process_video(video, output_path)

            # Integrate results into game
            self._integrate_video_analysis(game, analyzed_video, video_path)

            self.logger.info("Video analysis completed and integrated into game")
            return game

        except Exception as e:
            self.logger.error(f"Video analysis failed: {e}")
            raise

    def create_game_from_video(
        self, video_path: str, home_team: str, away_team: str, **kwargs
    ) -> Game:
        """
        Create a new Game object from video analysis.

        This is a convenience method that creates a game and analyzes video in one step.

        Args:
            video_path: Path to video file
            home_team: Home team name
            away_team: Away team name
            **kwargs: Additional arguments for game creation

        Returns:
            Game object with video analysis integrated
        """
        from ...game.game_factory import GameFactory

        # Create game
        game = GameFactory.create_from_video(
            video_path=video_path,
            home_team_name=home_team,
            away_team_name=away_team,
            **kwargs,
        )

        # Analyze video
        return self.analyze_video_for_game(game, video_path)

    def _integrate_video_analysis(
        self, game: Game, video: Video, video_path: str
    ) -> None:
        """
        Integrate video analysis results into a Game object.

        Args:
            game: Game to integrate results into
            video: Analyzed video data
            video_path: Original video file path
        """
        self.logger.info("Integrating video analysis into game")

        # Add analysis source
        import uuid

        source = AnalysisSource(
            source_id=str(uuid.uuid4()),
            source_type="video",
            description=f"Video analysis from {video_path}",
            metadata={
                "video_path": video_path,
                "video_duration": video.metadata.duration,
                "frame_rate": video.metadata.frame_rate,
                "resolution": video.metadata.resolution,
                "total_frames": len(video.frames),
            },
        )
        game.add_analysis_source(source)

        # Extract and integrate analysis data
        self._extract_player_data(game, video)
        self._extract_ball_data(game, video)
        self._extract_team_data(game, video)
        self._extract_events(game, video)

        # Update game metadata
        game.metadata.update(
            {
                "video_analysis_completed": True,
                "frames_analyzed": len(video.frames),
                "analysis_timestamp": video.metadata.duration,
            }
        )

    def _extract_player_data(self, game: Game, video: Video) -> None:
        """Extract player tracking and movement data from video analysis."""
        for frame in video.frames:
            # Process all players
            for track_id, player in frame.players.items():
                if player.player_id:
                    game_player = game.get_or_create_player(
                        player_id=player.player_id,
                        team_id=str(player.team_id) if player.team_id else None,
                        position=getattr(player, "position_role", None),
                        is_goalkeeper=False,
                    )

                    # Add position data if available
                    if player.pixel_position:
                        game_player.add_position_data(
                            timestamp=frame.timestamp,
                            x=player.pixel_position[0],
                            y=player.pixel_position[1],
                            velocity_x=getattr(player, "velocity_x", 0),
                            velocity_y=getattr(player, "velocity_y", 0),
                            speed=player.speed or 0,
                        )

            # Process goalkeepers
            for track_id, goalkeeper in frame.goalkeepers.items():
                if goalkeeper.player_id:
                    game_player = game.get_or_create_player(
                        player_id=goalkeeper.player_id,
                        team_id=str(goalkeeper.team_id) if goalkeeper.team_id else None,
                        position="GK",
                        is_goalkeeper=True,
                    )

                    # Add position data if available
                    if goalkeeper.pixel_position:
                        game_player.add_position_data(
                            timestamp=frame.timestamp,
                            x=goalkeeper.pixel_position[0],
                            y=goalkeeper.pixel_position[1],
                            velocity_x=getattr(goalkeeper, "velocity_x", 0),
                            velocity_y=getattr(goalkeeper, "velocity_y", 0),
                            speed=goalkeeper.speed or 0,
                        )

    def _extract_ball_data(self, game: Game, video: Video) -> None:
        """Extract ball tracking and possession data from video analysis."""
        for frame in video.frames:
            if frame.ball:
                ball = frame.ball
                # Add ball position
                if ball.pixel_position:
                    game.ball.add_position_data(
                        timestamp=frame.timestamp,
                        x=ball.pixel_position[0],
                        y=ball.pixel_position[1],
                        velocity_x=getattr(ball, "velocity_x", 0),
                        velocity_y=getattr(ball, "velocity_y", 0),
                    )

                # Add possession data if available
                if ball.controlling_player_id:
                    game.ball.add_possession_data(
                        timestamp=frame.timestamp,
                        player_id=str(ball.controlling_player_id),
                        team_id=(
                            str(ball.possession_team_id)
                            if ball.possession_team_id
                            else None
                        ),
                        confidence=ball.possession_confidence or 1.0,
                    )

    def _extract_team_data(self, game: Game, video: Video) -> None:
        """Extract team-level analysis data from video."""
        # Extract team colors and formation data from custom metadata
        team_analysis = video.custom.get("team_analysis", {})

        for team_id, team_data in team_analysis.items():
            team = game.get_team_by_id(team_id)
            if team:
                # Update team colors
                if "primary_color" in team_data:
                    team.primary_color = team_data["primary_color"]
                if "secondary_color" in team_data:
                    team.secondary_color = team_data["secondary_color"]

                # Add formation data
                formations = team_data.get("formations", [])
                for formation in formations:
                    team.add_formation_data(
                        timestamp=formation.get("timestamp"),
                        formation=formation.get("formation"),
                        players=formation.get("players", []),
                    )

    def _extract_events(self, game: Game, video: Video) -> None:
        """Extract match events from video analysis."""
        # Extract events from custom metadata
        events = video.custom.get("detected_events", [])

        for event_data in events:
            event = game.add_event(
                event_data.get("type", "unknown"),
                timestamp=event_data.get("timestamp"),
                location=event_data.get("location"),
                players_involved=event_data.get("players", []),
                team_id=event_data.get("team_id"),
                metadata=event_data.get("metadata", {}),
            )

    def get_analysis_summary(self, game: Game) -> Dict[str, Any]:
        """
        Generate analysis summary for a game with video analysis.

        Args:
            game: Game object to summarize

        Returns:
            Dictionary with analysis summary
        """
        video_sources = [
            source
            for source in game.analysis_sources.values()
            if source.source_type == "video"
        ]

        if not video_sources:
            return {"error": "No video analysis data found"}

        summary = {
            "game_info": {
                "game_id": game.game_id,
                "home_team": game.home_team.name,
                "away_team": game.away_team.name,
                "status": game.status.value,
                "video_sources": len(video_sources),
            },
            "analysis_summary": {
                "players_tracked": len(game.get_all_players()),
                "total_events": len(game.events),
                "analysis_sources": len(game.analysis_sources),
                "ball_tracking_points": len(game.ball.position_history),
                "possession_changes": len(getattr(game.ball, "possession_history", [])),
            },
            "video_summary": {},
        }

        # Add video-specific summary
        if video_sources:
            latest_video = max(video_sources, key=lambda x: len(x.metadata))
            summary["video_summary"] = {
                "video_path": latest_video.metadata.get("video_path"),
                "duration": latest_video.metadata.get("video_duration"),
                "frame_rate": latest_video.metadata.get("frame_rate"),
                "resolution": latest_video.metadata.get("resolution"),
                "frames_analyzed": latest_video.metadata.get("total_frames"),
            }

        return summary
