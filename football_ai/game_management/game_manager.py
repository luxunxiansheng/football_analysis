"""
Game Manager - High-level management of Game instances and analysis workflows.

This module provides the main interface for working with Games in the
game-centric architecture, supporting multiple analysis sources and
comprehensive match analytics.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import pickle

from ..domain.game import Game, AnalysisSource
from .factory import GameFactory


class GameManager:
    """
    Central manager for Game instances and comprehensive match analysis.

    Provides high-level APIs for creating, managing, and analyzing games
    from multiple data sources including video, GPS, manual annotations,
    and future analysis sources.
    """

    def __init__(self):
        self.games: Dict[str, Game] = {}
        self.active_game: Optional[Game] = None

    # Game lifecycle management
    def create_game_from_video(
        self, video_path: str, home_team: str, away_team: str, **kwargs
    ) -> Game:
        """Create a new game for video analysis."""
        game = GameFactory.create_from_video(
            video_path=video_path,
            home_team_name=home_team,
            away_team_name=away_team,
            **kwargs,
        )

        self.games[game.game_id] = game
        self.active_game = game

        return game

    def create_game_from_info(self, match_info: Dict[str, Any]) -> Game:
        """Create a game from structured match information."""
        game = GameFactory.create_from_match_info(match_info)
        self.games[game.game_id] = game
        self.active_game = game
        return game

    def load_game(self, game_id: str) -> Optional[Game]:
        """Load a game by ID."""
        return self.games.get(game_id)

    def set_active_game(self, game_id: str) -> bool:
        """Set the active game for operations."""
        if game_id in self.games:
            self.active_game = self.games[game_id]
            return True
        return False

    def get_active_game(self) -> Optional[Game]:
        """Get the currently active game."""
        return self.active_game

    # Analysis integration
    def add_video_analysis(
        self,
        video_analysis_results: Dict[str, Any],
        game_id: Optional[str] = None,
        source_id: str = "video_primary",
    ) -> bool:
        """
        Add video analysis results to a game.

        This is the main method for integrating video processing results.
        """
        game = self._get_target_game(game_id)
        if not game:
            return False

        game.integrate_video_analysis(video_analysis_results, source_id)
        return True

    def add_analysis_source(
        self, source: AnalysisSource, game_id: Optional[str] = None
    ) -> bool:
        """Add an analysis source to a game."""
        game = self._get_target_game(game_id)
        if not game:
            return False

        game.add_analysis_source(source)
        return True

    # Analysis and reporting
    def analyze_game(
        self, game_id: Optional[str] = None, analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive analysis on a game.

        Returns analysis results including possession, formations, events, etc.
        """
        game = self._get_target_game(game_id)
        if not game:
            return {}

        if analysis_types is None:
            analysis_types = ["possession", "summary", "heatmaps"]

        results = {}

        if "possession" in analysis_types:
            results["possession"] = game.calculate_possession()

        if "summary" in analysis_types:
            results["summary"] = game.get_match_summary()

        if "heatmaps" in analysis_types:
            results["heatmaps"] = self._generate_heatmaps(game)

        if "formations" in analysis_types:
            results["formations"] = self._analyze_formations(game)

        # Store results in game analytics
        game.analytics.update(results)

        return results

    def get_game_report(self, game_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive game report."""
        game = self._get_target_game(game_id)
        if not game:
            return {}

        return {
            "game_info": game.get_match_summary(),
            "analysis_sources": [
                {
                    "source_id": source.source_id,
                    "type": source.source_type,
                    "description": source.description,
                    "confidence": source.confidence,
                }
                for source in game.analysis_sources.values()
            ],
            "analytics": game.analytics,
            "events": [
                {
                    "type": event.event_type,
                    "time": event.match_time,
                    "description": event.description,
                }
                for event in game.events
            ],
        }

    # Persistence
    def save_game(self, game_id: str, filepath: str, format: str = "json") -> bool:
        """Save a game to file."""
        if game_id not in self.games:
            return False

        game = self.games[game_id]

        try:
            if format == "json":
                self._save_game_json(game, filepath)
            elif format == "pickle":
                self._save_game_pickle(game, filepath)
            else:
                return False
            return True
        except Exception:
            return False

    def load_game_from_file(
        self, filepath: str, format: str = "json"
    ) -> Optional[Game]:
        """Load a game from file."""
        try:
            if format == "json":
                game = self._load_game_json(filepath)
            elif format == "pickle":
                game = self._load_game_pickle(filepath)
            else:
                return None

            if game:
                self.games[game.game_id] = game
                return game
            return None
        except Exception:
            return None

    # Utility methods
    def list_games(self) -> List[Dict[str, Any]]:
        """List all loaded games with basic info."""
        return [
            {
                "game_id": game.game_id,
                "teams": f"{game.home_team.name} vs {game.away_team.name}",
                "date": game.match_date.isoformat(),
                "status": game.status.value,
                "sources": len(game.analysis_sources),
            }
            for game in self.games.values()
        ]

    def _get_target_game(self, game_id: Optional[str]) -> Optional[Game]:
        """Get target game (specified or active)."""
        if game_id:
            return self.games.get(game_id)
        return self.active_game

    def _generate_heatmaps(self, game: Game) -> Dict[str, Any]:
        """Generate heatmap data for all players."""
        heatmaps = {}

        for player in game.get_all_players():
            player_id = str(player.track_id)
            positions = game.get_player_heatmap_data(player_id)
            if positions:
                heatmaps[player_id] = {
                    "positions": positions,
                    "count": len(positions),
                    "team_id": player.team_id,
                }

        return heatmaps

    def _analyze_formations(self, game: Game) -> Dict[str, Any]:
        """Analyze team formations throughout the match."""
        # Placeholder for formation analysis
        # Would implement actual formation detection logic here
        return {"home_formations": [], "away_formations": [], "formation_changes": []}

    def _save_game_json(self, game: Game, filepath: str) -> None:
        """Save game as JSON (simplified representation)."""
        game_data = {
            "game_id": game.game_id,
            "home_team": {
                "team_id": game.home_team.team_id,
                "name": game.home_team.name,
            },
            "away_team": {
                "team_id": game.away_team.team_id,
                "name": game.away_team.name,
            },
            "match_date": game.match_date.isoformat(),
            "competition": game.competition,
            "venue": game.venue,
            "status": game.status.value,
            "final_score": game.final_score,
            "analytics": game.analytics,
        }

        with open(filepath, "w") as f:
            json.dump(game_data, f, indent=2, default=str)

    def _load_game_json(self, filepath: str) -> Optional[Game]:
        """Load game from JSON."""
        # Placeholder - would implement full JSON deserialization
        return None

    def _save_game_pickle(self, game: Game, filepath: str) -> None:
        """Save game as pickle (full object)."""
        with open(filepath, "wb") as f:
            pickle.dump(game, f)

    def _load_game_pickle(self, filepath: str) -> Optional[Game]:
        """Load game from pickle."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
