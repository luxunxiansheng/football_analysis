"""
Game Domain Model - The core entity representing a football match.

This is the central model that all analysis revolves around. A Game represents
the actual football match with its intrinsic properties, independent of how
we observe or analyze it (video, GPS, manual annotation, etc.).
"""

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from datetime import datetime, date
import uuid
from enum import Enum

# Import specialized domain models
from .player import Player
from .referee import Referee
from .goalkeeper import Goalkeeper
from .ball import Ball
from .field import Field


class MatchType(Enum):
    """Types of football matches."""

    LEAGUE = "league"
    CUP = "cup"
    FRIENDLY = "friendly"
    TRAINING = "training"
    PLAYOFF = "playoff"
    INTERNATIONAL = "international"


class MatchStatus(Enum):
    """Status of the match."""

    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


@dataclass
class Team:
    """Represents a football team."""

    team_id: str
    name: str
    short_name: Optional[str] = None
    country: Optional[str] = None
    league: Optional[str] = None

    # Squad for this match
    players: List[Player] = dataclass_field(default_factory=list)
    goalkeepers: List[Goalkeeper] = dataclass_field(default_factory=list)

    # Team visual identification
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None

    def add_player(self, player: Union[Player, Goalkeeper]) -> None:
        """Add a player to the team squad."""
        if isinstance(player, Goalkeeper):
            self.goalkeepers.append(player)
        else:
            self.players.append(player)

    def get_all_players(self) -> List[Union[Player, Goalkeeper]]:
        """Get all players including goalkeepers."""
        return list(self.players) + list(self.goalkeepers)

    def get_player_by_id(self, player_id: str) -> Optional[Union[Player, Goalkeeper]]:
        """Get a player by their ID."""
        for player in self.get_all_players():
            if player.player_id == player_id:
                return player
        return None

    def add_formation_data(
        self, timestamp: float, formation: str, players: List[str]
    ) -> None:
        """Add formation data for the team."""
        # Initialize formation_history if it doesn't exist
        if not hasattr(self, "formation_history"):
            self.formation_history = []

        self.formation_history.append(
            {"timestamp": timestamp, "formation": formation, "players": players}
        )


@dataclass
class MatchEvent:
    """Represents an event that occurred during the match."""

    event_id: str
    event_type: str  # "goal", "card", "substitution", "corner", etc.
    match_time: float  # Time in minutes from match start
    team_id: Optional[str] = None
    player_id: Optional[str] = None
    description: Optional[str] = None
    position: Optional[Tuple[float, float]] = None  # Field position if relevant
    additional_data: Dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass
class AnalysisSource:
    """Represents a source of analysis data for the game."""

    source_id: str
    source_type: str  # "video", "gps", "manual", "official", etc.
    description: Optional[str] = None
    confidence: float = 1.0
    coverage_start: Optional[float] = None  # Match time coverage start
    coverage_end: Optional[float] = None  # Match time coverage end
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass
class Game:
    """
    The core Game model representing a football match.

    This is the central entity around which all analysis revolves. A Game contains
    the essential match information and serves as the integration point for all
    analysis sources (video, GPS, manual annotations, etc.).
    """

    # Core match identification
    game_id: str
    home_team: Team
    away_team: Team
    match_date: date

    # Match context
    competition: Optional[str] = None
    season: Optional[str] = None
    matchday: Optional[int] = None
    venue: Optional[str] = None
    match_type: MatchType = MatchType.LEAGUE

    # Match state
    status: MatchStatus = MatchStatus.SCHEDULED
    kickoff_time: Optional[datetime] = None
    duration: float = 90.0  # Match duration in minutes

    # Officials
    referees: List[Referee] = dataclass_field(default_factory=list)

    # Field configuration
    field: Field = dataclass_field(default_factory=Field)

    # Match events and timeline
    events: List[MatchEvent] = dataclass_field(default_factory=list)

    # Analysis sources
    analysis_sources: Dict[str, AnalysisSource] = dataclass_field(default_factory=dict)

    # Temporal data - observations at specific match times
    # Structure: {match_time: {player_id: observation_data}}
    player_observations: Dict[float, Dict[str, Dict[str, Any]]] = dataclass_field(
        default_factory=dict
    )
    ball_observations: Dict[float, Dict[str, Any]] = dataclass_field(
        default_factory=dict
    )

    # Match-level analytics and results
    final_score: Optional[Tuple[int, int]] = None  # (home, away)
    half_time_score: Optional[Tuple[int, int]] = None
    analytics: Dict[str, Any] = dataclass_field(default_factory=dict)

    # External identifiers for integration
    external_ids: Dict[str, str] = dataclass_field(default_factory=dict)

    # Game metadata and configuration
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self):
        """Initialize game-specific data after creation."""
        if not self.game_id:
            # Generate a unique game ID if not provided
            date_str = self.match_date.strftime("%Y%m%d")
            self.game_id = (
                f"{date_str}_{self.home_team.team_id}_vs_{self.away_team.team_id}"
            )

    # Analysis source management
    def add_analysis_source(self, source: AnalysisSource) -> None:
        """Add an analysis source to the game."""
        self.analysis_sources[source.source_id] = source

    def get_analysis_source(self, source_id: str) -> Optional[AnalysisSource]:
        """Get an analysis source by ID."""
        return self.analysis_sources.get(source_id)

    def remove_analysis_source(self, source_id: str) -> bool:
        """Remove an analysis source."""
        if source_id in self.analysis_sources:
            del self.analysis_sources[source_id]
            return True
        return False

    # Player and team management
    def get_all_players(self) -> List[Union[Player, Goalkeeper]]:
        """Get all players from both teams."""
        return self.home_team.get_all_players() + self.away_team.get_all_players()

    def get_player_by_id(self, player_id: str) -> Optional[Union[Player, Goalkeeper]]:
        """Get a player by their ID from either team."""
        player = self.home_team.get_player_by_id(player_id)
        if player:
            return player
        return self.away_team.get_player_by_id(player_id)

    def get_team_by_id(self, team_id: str) -> Optional[Team]:
        """Get a team by ID."""
        if self.home_team.team_id == team_id:
            return self.home_team
        elif self.away_team.team_id == team_id:
            return self.away_team
        return None

    # Event management
    def add_event(self, event: Union[MatchEvent, str], **kwargs) -> MatchEvent:
        """
        Add an event to the match timeline.

        Args:
            event: Either a MatchEvent object or an event_type string
            **kwargs: Additional parameters when creating from event_type

        Returns:
            The added MatchEvent object
        """
        if isinstance(event, str):
            # Create MatchEvent from parameters
            import uuid

            match_event = MatchEvent(
                event_id=kwargs.get("event_id", str(uuid.uuid4())),
                event_type=event,
                match_time=kwargs.get("timestamp", 0.0),
                team_id=kwargs.get("team_id"),
                player_id=(
                    kwargs.get("players_involved", [None])[0]
                    if kwargs.get("players_involved")
                    else None
                ),
                description=kwargs.get("description"),
                position=kwargs.get("location"),
                additional_data=kwargs.get("metadata", {}),
            )
        else:
            match_event = event

        self.events.append(match_event)
        # Keep events sorted by match time
        self.events.sort(key=lambda e: e.match_time)
        return match_event

    def get_events_by_type(self, event_type: str) -> List[MatchEvent]:
        """Get all events of a specific type."""
        return [event for event in self.events if event.event_type == event_type]

    def get_events_in_time_range(
        self, start_time: float, end_time: float
    ) -> List[MatchEvent]:
        """Get events within a specific time range."""
        return [
            event for event in self.events if start_time <= event.match_time <= end_time
        ]

    # Observation data management
    def add_player_observation(
        self,
        match_time: float,
        player_id: str,
        observation: Dict[str, Any],
        source_id: str,
    ) -> None:
        """Add an observation for a player at a specific match time."""
        if match_time not in self.player_observations:
            self.player_observations[match_time] = {}

        if player_id not in self.player_observations[match_time]:
            self.player_observations[match_time][player_id] = {}

        # Add source attribution
        observation["source"] = source_id
        observation["timestamp"] = match_time

        # Store the observation (could have multiple sources for same time)
        source_key = f"{source_id}_data"
        self.player_observations[match_time][player_id][source_key] = observation

    def add_ball_observation(
        self, match_time: float, observation: Dict[str, Any], source_id: str
    ) -> None:
        """Add a ball observation at a specific match time."""
        if match_time not in self.ball_observations:
            self.ball_observations[match_time] = {}

        # Add source attribution
        observation["source"] = source_id
        observation["timestamp"] = match_time

        source_key = f"{source_id}_data"
        self.ball_observations[match_time][source_key] = observation

    def get_player_observations_in_range(
        self, player_id: str, start_time: float, end_time: float
    ) -> Dict[float, Dict[str, Any]]:
        """Get all observations for a player within a time range."""
        observations = {}
        for match_time, time_data in self.player_observations.items():
            if start_time <= match_time <= end_time and player_id in time_data:
                observations[match_time] = time_data[player_id]
        return observations

    def get_ball_observations_in_range(
        self, start_time: float, end_time: float
    ) -> Dict[float, Dict[str, Any]]:
        """Get all ball observations within a time range."""
        observations = {}
        for match_time, time_data in self.ball_observations.items():
            if start_time <= match_time <= end_time:
                observations[match_time] = time_data
        return observations

    # Analysis and analytics methods
    def calculate_possession(
        self, start_time: float = 0.0, end_time: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate ball possession percentages for each team."""
        if end_time is None:
            end_time = self.duration

        ball_obs = self.get_ball_observations_in_range(start_time, end_time)

        team_possession = {self.home_team.team_id: 0, self.away_team.team_id: 0}
        total_observations = 0

        for match_time, observations in ball_obs.items():
            for source_key, obs_data in observations.items():
                if "possession_team" in obs_data:
                    team_id = obs_data["possession_team"]
                    if team_id in team_possession:
                        team_possession[team_id] += 1
                    total_observations += 1

        # Convert to percentages
        if total_observations > 0:
            return {
                team_id: (count / total_observations) * 100
                for team_id, count in team_possession.items()
            }
        return {team_id: 0.0 for team_id in team_possession.keys()}

    def get_player_heatmap_data(
        self, player_id: str, source_id: Optional[str] = None
    ) -> List[Tuple[float, float]]:
        """Get position data for creating player heatmaps."""
        positions = []

        for match_time, time_data in self.player_observations.items():
            if player_id in time_data:
                player_data = time_data[player_id]

                # Filter by source if specified
                sources_to_check = [source_id] if source_id else player_data.keys()

                for source_key in sources_to_check:
                    if source_key in player_data:
                        obs = player_data[source_key]
                        if "field_position" in obs and obs["field_position"]:
                            positions.append(obs["field_position"])

        return positions

    def get_match_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the match."""
        return {
            "game_id": self.game_id,
            "teams": {
                "home": {
                    "name": self.home_team.name,
                    "players": len(self.home_team.get_all_players()),
                },
                "away": {
                    "name": self.away_team.name,
                    "players": len(self.away_team.get_all_players()),
                },
            },
            "match_info": {
                "date": self.match_date.isoformat(),
                "competition": self.competition,
                "venue": self.venue,
                "status": self.status.value,
                "duration": self.duration,
            },
            "score": {"final": self.final_score, "half_time": self.half_time_score},
            "analysis": {
                "sources": len(self.analysis_sources),
                "events": len(self.events),
                "player_observations": len(self.player_observations),
                "ball_observations": len(self.ball_observations),
            },
            "possession": self.calculate_possession(),
        }

    # Video analysis integration methods
    def integrate_video_analysis(
        self, video_analysis_results, source_id: str = "video_primary"
    ) -> None:
        """
        Integrate results from video analysis into the game.

        This method takes video analysis results and converts them into
        game observations with proper time mapping.
        """
        # Add the video as an analysis source
        video_source = AnalysisSource(
            source_id=source_id,
            source_type="video",
            description="Video analysis results",
            metadata=video_analysis_results.get("metadata", {}),
        )
        self.add_analysis_source(video_source)

        # Process frame-by-frame observations
        if "frames" in video_analysis_results:
            for frame_data in video_analysis_results["frames"]:
                match_time = self._video_time_to_match_time(frame_data["timestamp"])

                # Add player observations
                if "players" in frame_data:
                    for player_data in frame_data["players"]:
                        observation = {
                            "pixel_position": player_data.get("pixel_position"),
                            "field_position": player_data.get("field_position"),
                            "team_id": player_data.get("team_id"),
                            "confidence": player_data.get("confidence", 1.0),
                        }
                        self.add_player_observation(
                            match_time,
                            str(player_data["track_id"]),
                            observation,
                            source_id,
                        )

                # Add ball observations
                if "ball" in frame_data and frame_data["ball"]:
                    ball_data = frame_data["ball"]
                    observation = {
                        "pixel_position": ball_data.get("pixel_position"),
                        "field_position": ball_data.get("field_position"),
                        "possession_team": ball_data.get("possession_team_id"),
                        "controlling_player": ball_data.get("controlling_player_id"),
                        "confidence": ball_data.get("confidence", 1.0),
                    }
                    self.add_ball_observation(match_time, observation, source_id)

    def _video_time_to_match_time(self, video_timestamp: float) -> float:
        """
        Convert video timestamp to match time.

        For now, assumes video starts at match start (0:00).
        In future, this could be more sophisticated with calibration.
        """
        # Simple 1:1 mapping for now - video seconds to match minutes
        return video_timestamp / 60.0

    # Utility methods
    def __str__(self) -> str:
        """String representation of the game."""
        return (
            f"Game({self.home_team.name} vs {self.away_team.name}, {self.match_date})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Game(id='{self.game_id}', "
            f"teams='{self.home_team.name} vs {self.away_team.name}', "
            f"date='{self.match_date}', status='{self.status.value}')"
        )

    def get_or_create_player(
        self,
        player_id: str,
        team_id: Optional[str] = None,
        position: Optional[str] = None,
        is_goalkeeper: bool = False,
    ) -> Union[Player, Goalkeeper]:
        """
        Get an existing player or create a new one if not found.

        Args:
            player_id: Unique identifier for the player
            team_id: Team ID (if creating new player)
            position: Player position (if creating new player)
            is_goalkeeper: Whether player is a goalkeeper

        Returns:
            Player or Goalkeeper object
        """
        # Try to find existing player
        existing_player = self.get_player_by_id(player_id)
        if existing_player:
            return existing_player

        # Generate a unique track_id based on player_id
        track_id = hash(player_id) % 100000  # Simple hash to numeric ID

        # Create new player
        if is_goalkeeper:
            from .goalkeeper import Goalkeeper

            player = Goalkeeper(
                track_id=track_id,
                player_id=player_id,
                jersey_number=None,
                team_id=int(team_id) if team_id and team_id.isdigit() else None,
            )
        else:
            from .player import Player

            player = Player(
                track_id=track_id,
                player_id=player_id,
                jersey_number=None,
                team_id=int(team_id) if team_id and team_id.isdigit() else None,
            )

        # Add to appropriate team
        if team_id == self.home_team.team_id:
            self.home_team.add_player(player)
        elif team_id == self.away_team.team_id:
            self.away_team.add_player(player)
        else:
            # If no team specified or team not found, add to home team as default
            self.home_team.add_player(player)

        return player

    @property
    def ball(self) -> "Ball":
        """Get the ball object for this game."""
        if not hasattr(self, "_ball"):
            from .ball import Ball  # Import here to avoid circular imports

            self._ball = Ball(track_id=0)  # Use track_id 0 for the ball
        return self._ball
