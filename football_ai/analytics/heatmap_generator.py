"""
Heatmap Generator

Generates visual heatmaps for player positioning, team formations,
and match intensity visualization.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from ..core_models.game import Game


class HeatmapGenerator:
    """
    Generates various types of heatmaps for match analysis visualization.

    This provides high-level visualization data for:
    - Player positioning heatmaps
    - Team formation heatmaps
    - Ball movement heatmaps
    - Match intensity maps
    """

    def __init__(self, field_width: float = 105.0, field_height: float = 68.0):
        self.field_width = field_width
        self.field_height = field_height
        self.grid_resolution = (50, 34)  # Grid cells for heatmap

    def generate_player_heatmap(self, game: Game, player_id: str) -> Dict[str, Any]:
        """
        Generate positioning heatmap for a specific player.

        Args:
            game: Game object with player tracking data
            player_id: ID of player to generate heatmap for

        Returns:
            Heatmap data with coordinates and intensity values
        """
        player_positions = game.get_player_heatmap_data(player_id)

        if not player_positions:
            return {"error": f"No position data found for player {player_id}"}

        # Convert positions to heatmap grid
        heatmap_grid = self._positions_to_grid(player_positions)

        return {
            "player_id": player_id,
            "heatmap_data": heatmap_grid.tolist(),
            "grid_resolution": self.grid_resolution,
            "field_dimensions": (self.field_width, self.field_height),
            "total_positions": len(player_positions),
            "coverage_area": self._calculate_coverage_area(heatmap_grid),
            "hotspots": self._find_hotspots(heatmap_grid),
        }

    def generate_team_heatmap(self, game: Game, team_id: str) -> Dict[str, Any]:
        """
        Generate combined heatmap for all players in a team.

        Args:
            game: Game object
            team_id: Team identifier

        Returns:
            Team-level heatmap data
        """
        team_players = game.get_team_players(team_id)
        combined_grid = np.zeros(self.grid_resolution)

        for player in team_players:
            player_positions = game.get_player_heatmap_data(str(player.track_id))
            if player_positions:
                player_grid = self._positions_to_grid(player_positions)
                combined_grid += player_grid

        # Normalize by number of players
        if len(team_players) > 0:
            combined_grid /= len(team_players)

        return {
            "team_id": team_id,
            "heatmap_data": combined_grid.tolist(),
            "grid_resolution": self.grid_resolution,
            "field_dimensions": (self.field_width, self.field_height),
            "player_count": len(team_players),
            "formation_compactness": self._calculate_compactness(combined_grid),
            "defensive_line": self._find_defensive_line(combined_grid),
            "attacking_focus": self._find_attacking_focus(combined_grid),
        }

    def generate_ball_heatmap(self, game: Game) -> Dict[str, Any]:
        """
        Generate heatmap showing ball movement and concentration areas.

        Args:
            game: Game object with ball tracking data

        Returns:
            Ball movement heatmap data
        """
        ball_positions = game.ball.position_history

        if not ball_positions:
            return {"error": "No ball position data found"}

        # Convert ball positions to grid format
        positions_for_grid = [{"x": pos.x, "y": pos.y} for pos in ball_positions]

        heatmap_grid = self._positions_to_grid(positions_for_grid)

        return {
            "heatmap_data": heatmap_grid.tolist(),
            "grid_resolution": self.grid_resolution,
            "field_dimensions": (self.field_width, self.field_height),
            "total_ball_positions": len(ball_positions),
            "most_active_zones": self._find_hotspots(heatmap_grid),
            "possession_centers": self._find_possession_centers(heatmap_grid),
        }

    def generate_match_intensity_heatmap(self, game: Game) -> Dict[str, Any]:
        """
        Generate heatmap showing match intensity across different field areas.

        Args:
            game: Game object

        Returns:
            Match intensity heatmap data
        """
        # Combine player activity, ball activity, and events
        intensity_grid = np.zeros(self.grid_resolution)

        # Add player movement intensity
        for player in game.get_all_players():
            player_positions = game.get_player_heatmap_data(str(player.track_id))
            if player_positions:
                # Weight by player speed/activity
                weighted_positions = self._weight_by_activity(player_positions, player)
                player_grid = self._positions_to_grid(weighted_positions)
                intensity_grid += player_grid

        # Add ball activity
        ball_positions = game.ball.position_history
        if ball_positions:
            positions_for_grid = [{"x": pos.x, "y": pos.y} for pos in ball_positions]
            ball_grid = self._positions_to_grid(positions_for_grid)
            intensity_grid += ball_grid * 2  # Weight ball activity higher

        return {
            "intensity_data": intensity_grid.tolist(),
            "grid_resolution": self.grid_resolution,
            "field_dimensions": (self.field_width, self.field_height),
            "peak_intensity_zones": self._find_hotspots(intensity_grid),
            "quiet_zones": self._find_quiet_zones(intensity_grid),
            "average_intensity": float(np.mean(intensity_grid)),
        }

    def _positions_to_grid(self, positions: List[Dict[str, float]]) -> np.ndarray:
        """Convert position coordinates to heatmap grid."""
        grid = np.zeros(self.grid_resolution)

        for pos in positions:
            # Convert field coordinates to grid indices
            x_idx = int((pos["x"] / self.field_width) * self.grid_resolution[0])
            y_idx = int((pos["y"] / self.field_height) * self.grid_resolution[1])

            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, self.grid_resolution[0] - 1))
            y_idx = max(0, min(y_idx, self.grid_resolution[1] - 1))

            grid[x_idx, y_idx] += 1

        return grid

    def _calculate_coverage_area(self, grid: np.ndarray) -> float:
        """Calculate the percentage of field covered by player."""
        non_zero_cells = np.count_nonzero(grid)
        total_cells = grid.size
        return (non_zero_cells / total_cells) * 100

    def _find_hotspots(
        self, grid: np.ndarray, threshold_percentile: int = 90
    ) -> List[Dict[str, Any]]:
        """Find the most active areas (hotspots) in the heatmap."""
        threshold = np.percentile(grid, threshold_percentile)
        hotspot_indices = np.where(grid >= threshold)

        hotspots = []
        for i, j in zip(hotspot_indices[0], hotspot_indices[1]):
            # Convert grid indices back to field coordinates
            x = (i / self.grid_resolution[0]) * self.field_width
            y = (j / self.grid_resolution[1]) * self.field_height

            hotspots.append(
                {
                    "x": x,
                    "y": y,
                    "intensity": float(grid[i, j]),
                    "grid_position": (i, j),
                }
            )

        return hotspots

    def _calculate_compactness(self, grid: np.ndarray) -> float:
        """Calculate team formation compactness score."""
        # Calculate spread of team positions
        center_of_mass = self._find_center_of_mass(grid)
        weighted_distances = []

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] > 0:
                    distance = np.sqrt(
                        (i - center_of_mass[0]) ** 2 + (j - center_of_mass[1]) ** 2
                    )
                    weighted_distances.append(distance * grid[i, j])

        return float(np.mean(weighted_distances)) if weighted_distances else 0.0

    def _find_center_of_mass(self, grid: np.ndarray) -> Tuple[float, float]:
        """Find the center of mass of the heatmap."""
        total_mass = np.sum(grid)
        if total_mass == 0:
            return (0, 0)

        x_center = np.sum(np.arange(grid.shape[0])[:, np.newaxis] * grid) / total_mass
        y_center = np.sum(np.arange(grid.shape[1])[np.newaxis, :] * grid) / total_mass

        return (float(x_center), float(y_center))

    def _find_defensive_line(self, grid: np.ndarray) -> Dict[str, float]:
        """Find the average defensive line position."""
        # Find the x-coordinate with highest activity in defensive areas
        defensive_third = grid[: int(grid.shape[0] * 0.33), :]
        avg_defensive_line = (
            np.mean(np.where(defensive_third > 0)[0])
            if np.any(defensive_third > 0)
            else 0
        )

        return {
            "x_position": float(avg_defensive_line),
            "field_percentage": (avg_defensive_line / grid.shape[0]) * 100,
        }

    def _find_attacking_focus(self, grid: np.ndarray) -> Dict[str, float]:
        """Find the center of attacking focus."""
        attacking_third = grid[int(grid.shape[0] * 0.67) :, :]
        center_of_mass = self._find_center_of_mass(attacking_third)

        return {
            "x_position": float(center_of_mass[0]) + int(grid.shape[0] * 0.67),
            "y_position": float(center_of_mass[1]),
            "intensity": (
                float(np.max(attacking_third)) if attacking_third.size > 0 else 0
            ),
        }

    def _find_possession_centers(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Find main ball possession centers."""
        return self._find_hotspots(grid, threshold_percentile=85)

    def _weight_by_activity(
        self, positions: List[Dict[str, float]], player
    ) -> List[Dict[str, float]]:
        """Weight positions by player activity level (speed, etc.)."""
        # Simple implementation - could be enhanced with actual speed data
        return positions

    def _find_quiet_zones(
        self, grid: np.ndarray, threshold_percentile: int = 10
    ) -> List[Dict[str, Any]]:
        """Find areas with low activity."""
        threshold = np.percentile(grid, threshold_percentile)
        quiet_indices = np.where(grid <= threshold)

        quiet_zones = []
        for i, j in zip(quiet_indices[0], quiet_indices[1]):
            x = (i / self.grid_resolution[0]) * self.field_width
            y = (j / self.grid_resolution[1]) * self.field_height

            quiet_zones.append(
                {
                    "x": x,
                    "y": y,
                    "intensity": float(grid[i, j]),
                    "grid_position": (i, j),
                }
            )

        return quiet_zones
