"""
Ball Possession Analysis Module

This module provides functionality to analyze ball possession by tracking
the ball's proximity to players and determining team control.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import deque, Counter

from ..domain.models import Detection, PlayerState, TeamAssignment
from ..domain.interfaces import BallPossessionAnalyzer


class DistanceBasedBallPossessionAnalyzer(BallPossessionAnalyzer):
    """
    Implementation of ball possession analysis using distance-based
    assignment and temporal smoothing for stable possession detection.
    """

    def __init__(
        self,
        possession_distance: float = 70.0,
        confidence_threshold: int = 3,
        history_length: int = 5,
    ):
        """
        Initialize the ball possession analyzer.

        Args:
            possession_distance: Maximum distance for ball possession (pixels)
            confidence_threshold: Minimum frames needed to confirm possession
            history_length: Number of frames to consider for smoothing
        """
        self.possession_distance = possession_distance
        self.confidence_threshold = confidence_threshold
        self.history_length = history_length

        # Tracking state
        self._possession_history: deque = deque(maxlen=history_length)
        self._player_ball_distances: Dict[int, List[float]] = {}
        self._current_possessor: Optional[int] = None
        self._possession_confidence: int = 0

    def analyze_possession(
        self, ball_detections: List[Detection], player_states: List[PlayerState]
    ) -> Dict[str, Any]:
        """
        Analyze ball possession for the current frame.

        Args:
            ball_detections: List of ball detections (usually 0 or 1)
            player_states: List of player states with positions

        Returns:
            Dictionary containing possession analysis results
        """
        if not ball_detections or not player_states:
            return self._get_empty_result()

        # Use the most confident ball detection
        ball_detection = max(ball_detections, key=lambda x: x.confidence)
        ball_position = ball_detection.bbox.center

        # Calculate distances to all players
        player_distances = {}
        for player in player_states:
            if player.track_id is not None:
                player_position = (
                    player.bbox.foot_position
                )  # Use foot position for players
                distance = self._calculate_distance(ball_position, player_position)
                player_distances[player.track_id] = distance

                # Update distance history
                if player.track_id not in self._player_ball_distances:
                    self._player_ball_distances[player.track_id] = []
                self._player_ball_distances[player.track_id].append(distance)

                # Keep only recent history
                if (
                    len(self._player_ball_distances[player.track_id])
                    > self.history_length
                ):
                    self._player_ball_distances[player.track_id] = (
                        self._player_ball_distances[player.track_id][
                            -self.history_length :
                        ]
                    )

        # Find closest player
        closest_player_id = None
        min_distance = float("inf")

        for player_id, distance in player_distances.items():
            if distance < min_distance:
                min_distance = distance
                closest_player_id = player_id

        # Determine possession based on distance threshold
        has_possession = min_distance <= self.possession_distance

        # Update possession state with confidence tracking
        if has_possession and closest_player_id is not None:
            if self._current_possessor == closest_player_id:
                self._possession_confidence = min(
                    self._possession_confidence + 1, self.confidence_threshold + 2
                )
            else:
                if self._possession_confidence > 0:
                    self._possession_confidence -= 1
                else:
                    self._current_possessor = closest_player_id
                    self._possession_confidence = 1
        else:
            if self._possession_confidence > 0:
                self._possession_confidence -= 1
            else:
                self._current_possessor = None

        # Confirm possession only if confidence threshold is met
        confirmed_possessor = (
            self._current_possessor
            if self._possession_confidence >= self.confidence_threshold
            else None
        )

        # Update possession history
        self._possession_history.append(confirmed_possessor)

        # Assign ball possession to players
        for player in player_states:
            player.has_ball = player.track_id == confirmed_possessor

        # Calculate team possession statistics
        team_possession = self._calculate_team_possession(
            player_states, confirmed_possessor
        )

        return {
            "possessor_id": confirmed_possessor,
            "ball_position": ball_position,
            "min_distance": min_distance,
            "has_possession": confirmed_possessor is not None,
            "team_possession": team_possession,
            "possession_confidence": self._possession_confidence,
            "player_distances": player_distances,
        }

    def get_possession_stats(
        self, player_states_history: List[List[PlayerState]]
    ) -> Dict[str, float]:
        """
        Calculate overall possession statistics from match history.

        Args:
            player_states_history: History of player states for all frames

        Returns:
            Dictionary with possession statistics
        """
        team_1_frames = 0
        team_2_frames = 0
        total_frames = 0

        for frame_players in player_states_history:
            possessor = None
            for player in frame_players:
                if player.has_ball:
                    possessor = player
                    break

            if possessor and possessor.team:
                total_frames += 1
                if possessor.team == TeamAssignment.TEAM_1:
                    team_1_frames += 1
                elif possessor.team == TeamAssignment.TEAM_2:
                    team_2_frames += 1

        if total_frames == 0:
            return {
                "team_1_possession": 0.0,
                "team_2_possession": 0.0,
                "neutral_time": 100.0,
            }

        team_1_pct = (team_1_frames / total_frames) * 100
        team_2_pct = (team_2_frames / total_frames) * 100
        neutral_pct = 100 - team_1_pct - team_2_pct

        return {
            "team_1_possession": team_1_pct,
            "team_2_possession": team_2_pct,
            "neutral_time": neutral_pct,
        }

    def _calculate_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two positions."""
        return float(np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2))

    def _calculate_team_possession(
        self, player_states: List[PlayerState], possessor_id: Optional[int]
    ) -> Optional[TeamAssignment]:
        """Determine which team has possession based on the possessor."""
        if possessor_id is None:
            return None

        for player in player_states:
            if player.track_id == possessor_id:
                return player.team

        return None

    def _get_empty_result(self) -> Dict[str, Any]:
        """Return empty result when no ball or players are detected."""
        return {
            "possessor_id": None,
            "ball_position": None,
            "min_distance": float("inf"),
            "has_possession": False,
            "team_possession": None,
            "possession_confidence": 0,
            "player_distances": {},
        }

    def reset(self):
        """Reset the analyzer state."""
        self._possession_history.clear()
        self._player_ball_distances.clear()
        self._current_possessor = None
        self._possession_confidence = 0

    def get_current_possessor(self) -> Optional[int]:
        """Get the current ball possessor ID."""
        return self._current_possessor

    def get_possession_history(self) -> List[Optional[int]]:
        """Get the possession history."""
        return list(self._possession_history)
