import sys
from typing import Dict, List, Tuple

sys.path.append("../")
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    """
    Assigns a ball to the closest player based on distance calculations.

    This class determines which player is closest to the ball by measuring
    distances from the ball position to the left and right edges of each
    player's bounding box.
    """

    def __init__(self) -> None:
        """Initialize the PlayerBallAssigner with default maximum distance."""
        self.max_player_ball_distance: int = 70

    def assign_ball_to_player(
        self,
        players: Dict[int, Dict[str, List[float]]],
        ball_bbox: List[float],
    ) -> int:
        """
        Assign the ball to the closest player within maximum distance.

        Args:
            players: Dictionary mapping player IDs to player data containing 'bbox'
            ball_bbox: Bounding box coordinates of the ball [x1, y1, x2, y2]

        Returns:
            Player ID of the assigned player, or -1 if no player is close enough
        """
        ball_position: Tuple[float, float] = get_center_of_bbox(ball_bbox)

        minimum_distance: float = 99999
        assigned_player: int = -1

        for player_id, player in players.items():
            player_bbox: List[float] = player["bbox"]

            distance_left: float = measure_distance(
                (player_bbox[0], player_bbox[-1]), ball_position
            )
            distance_right: float = measure_distance(
                (player_bbox[2], player_bbox[-1]), ball_position
            )
            distance: float = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player
