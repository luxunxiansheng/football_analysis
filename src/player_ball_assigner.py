import sys
from typing import Dict, List, Tuple

sys.path.append("../")
from utils.bbox_utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    """
    Assigns a ball to the closest player based on distance calculations.

    This class determines which player is closest to the ball by measuring
    distances from the ball position to the left and right edges of each
    player's bounding box.
    """

    def __init__(self, max_player_ball_distance: int = 70) -> None:
        """Initialize the PlayerBallAssigner with default maximum distance."""
        self.max_player_ball_distance = max_player_ball_distance

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
        ball_position = get_center_of_bbox(ball_bbox)
        minimum_distance = float("inf")
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player["bbox"]

            # Calculate distance to left and right edges of player bbox
            distance_left = measure_distance(
                (player_bbox[0], player_bbox[-1]), ball_position
            )
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[-1]), ball_position
            )
            distance = min(distance_left, distance_right)

            # Assign ball to closest player within threshold
            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
