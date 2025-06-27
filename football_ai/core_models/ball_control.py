from dataclasses import dataclass
from typing import Optional


@dataclass
class BallControl:
    """Ball control information for a frame."""

    controlling_player: Optional[int] = None  # track_id of controlling player
    possession_team: Optional[int] = None  # team_id with possession
    control_confidence: Optional[float] = None  # confidence score 0-1
    last_touch_player: Optional[int] = None  # track_id of last player to touch ball

    def __bool__(self) -> bool:
        """Check if any ball control data is present."""
        return any(
            [
                self.controlling_player,
                self.possession_team,
                self.control_confidence,
                self.last_touch_player,
            ]
        )
