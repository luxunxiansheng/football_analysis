"""
Game Factory - Helper methods for creating Game instances from various sources.

This module provides convenient factory methods for creating Game objects
from different scenarios: video files, manual setup, external data, etc.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Optional, Dict, Any
import re

from ..domain.game import Game, Team, MatchType, MatchStatus
from ..domain import Player, Goalkeeper, Referee


class GameFactory:
    """Factory class for creating Game instances."""

    @staticmethod
    def create_from_video(
        video_path: str,
        home_team_name: str,
        away_team_name: str,
        match_date: Optional[date] = None,
        **kwargs,
    ) -> Game:
        """
        Create a Game instance for video analysis.

        This is the primary method for creating games when video is the main source.
        """
        if match_date is None:
            match_date = date.today()

        # Create basic teams
        home_team = Team(
            team_id=GameFactory._sanitize_team_id(home_team_name), name=home_team_name
        )
        away_team = Team(
            team_id=GameFactory._sanitize_team_id(away_team_name), name=away_team_name
        )

        # Generate game ID from video filename if not provided
        video_name = Path(video_path).stem
        game_id = kwargs.get("game_id", f"{match_date.strftime('%Y%m%d')}_{video_name}")

        game = Game(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            competition=kwargs.get("competition"),
            season=kwargs.get("season"),
            venue=kwargs.get("venue"),
            match_type=kwargs.get("match_type", MatchType.LEAGUE),
            status=MatchStatus.SCHEDULED,
            duration=kwargs.get("duration", 90.0),
        )

        return game

    @staticmethod
    def create_from_match_info(match_info: Dict[str, Any]) -> Game:
        """
        Create a Game instance from structured match information.

        Expected format:
        {
            'home_team': {'name': 'Barcelona', 'id': 'BAR'},
            'away_team': {'name': 'Real Madrid', 'id': 'MAD'},
            'date': '2025-06-24',
            'competition': 'La Liga',
            'venue': 'Camp Nou'
        }
        """
        # Parse date
        if isinstance(match_info["date"], str):
            match_date = datetime.strptime(match_info["date"], "%Y-%m-%d").date()
        else:
            match_date = match_info["date"]

        # Create teams
        home_info = match_info["home_team"]
        away_info = match_info["away_team"]

        home_team = Team(
            team_id=home_info.get(
                "id", GameFactory._sanitize_team_id(home_info["name"])
            ),
            name=home_info["name"],
            short_name=home_info.get("short_name"),
            country=home_info.get("country"),
            league=match_info.get("competition"),
        )

        away_team = Team(
            team_id=away_info.get(
                "id", GameFactory._sanitize_team_id(away_info["name"])
            ),
            name=away_info["name"],
            short_name=away_info.get("short_name"),
            country=away_info.get("country"),
            league=match_info.get("competition"),
        )

        game = Game(
            game_id=match_info.get("game_id", ""),
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            competition=match_info.get("competition"),
            season=match_info.get("season"),
            matchday=match_info.get("matchday"),
            venue=match_info.get("venue"),
            match_type=MatchType(match_info.get("match_type", "league")),
            status=MatchStatus(match_info.get("status", "scheduled")),
            duration=match_info.get("duration", 90.0),
        )

        # Add referees if provided
        if "referees" in match_info:
            for ref_info in match_info["referees"]:
                referee = Referee(
                    track_id=ref_info.get("track_id", 0),
                    referee_id=ref_info.get("id"),
                    referee_type=ref_info.get("role", "main"),
                )
                game.referees.append(referee)

        return game

    @staticmethod
    def create_quick_game(
        home_team: str, away_team: str, game_id: Optional[str] = None
    ) -> Game:
        """Create a quick game for testing or simple scenarios."""
        today = date.today()

        if game_id is None:
            game_id = f"{today.strftime('%Y%m%d')}_{home_team}_vs_{away_team}"

        home_team_obj = Team(
            team_id=GameFactory._sanitize_team_id(home_team), name=home_team
        )
        away_team_obj = Team(
            team_id=GameFactory._sanitize_team_id(away_team), name=away_team
        )

        return Game(
            game_id=game_id,
            home_team=home_team_obj,
            away_team=away_team_obj,
            match_date=today,
        )

    @staticmethod
    def _sanitize_team_id(team_name: str) -> str:
        """Convert team name to a valid team ID."""
        # Remove special characters and spaces, convert to uppercase
        team_id = re.sub(r"[^\w\s]", "", team_name)
        team_id = re.sub(r"\s+", "_", team_id)
        return team_id.upper()[:10]  # Limit to 10 characters
