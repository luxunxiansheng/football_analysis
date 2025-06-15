"""
Football Field Constants

This module contains standardized field dimensions for different types of soccer fields.
All dimensions are in meters.
"""

from typing import Dict, Tuple, Any


class FieldDimensions:
    """Standard field dimensions for different types of soccer fields."""

    # FIFA Standard Dimensions (International matches)
    FIFA_STANDARD = {
        "width": 68.0,
        "height": 105.0,
        "name": "FIFA Standard",
        "description": "Official FIFA standard dimensions for international matches",
    }

    # FIFA Minimum/Maximum allowed dimensions
    FIFA_MINIMUM = {
        "width": 45.0,
        "height": 90.0,
        "name": "FIFA Minimum",
        "description": "Minimum allowed dimensions per FIFA regulations",
    }

    FIFA_MAXIMUM = {
        "width": 90.0,
        "height": 120.0,
        "name": "FIFA Maximum",
        "description": "Maximum allowed dimensions per FIFA regulations",
    }

    # Premier League standard
    PREMIER_LEAGUE = {
        "width": 68.0,
        "height": 105.0,
        "name": "Premier League",
        "description": "Standard Premier League field dimensions",
    }

    # UEFA Champions League
    UEFA_CHAMPIONS = {
        "width": 68.0,
        "height": 105.0,
        "name": "UEFA Champions League",
        "description": "UEFA Champions League standard dimensions",
    }

    # La Liga standard
    LA_LIGA = {
        "width": 68.0,
        "height": 105.0,
        "name": "La Liga",
        "description": "Spanish La Liga standard dimensions",
    }

    # Bundesliga standard
    BUNDESLIGA = {
        "width": 68.0,
        "height": 105.0,
        "name": "Bundesliga",
        "description": "German Bundesliga standard dimensions",
    }

    # Serie A standard
    SERIE_A = {
        "width": 68.0,
        "height": 105.0,
        "name": "Serie A",
        "description": "Italian Serie A standard dimensions",
    }

    # Youth fields
    YOUTH_U12 = {
        "width": 45.0,
        "height": 64.0,
        "name": "Youth U12",
        "description": "Youth field for players under 12",
    }

    YOUTH_U14 = {
        "width": 55.0,
        "height": 75.0,
        "name": "Youth U14",
        "description": "Youth field for players under 14",
    }

    YOUTH_U16 = {
        "width": 64.0,
        "height": 91.0,
        "name": "Youth U16",
        "description": "Youth field for players under 16",
    }

    # 7-a-side and 5-a-side
    SEVEN_A_SIDE = {
        "width": 50.0,
        "height": 70.0,
        "name": "7-a-side",
        "description": "Standard 7-a-side field dimensions",
    }

    FIVE_A_SIDE = {
        "width": 25.0,
        "height": 42.0,
        "name": "5-a-side",
        "description": "Standard 5-a-side field dimensions",
    }

    # Women's football (same as men's for professional)
    WOMENS_PROFESSIONAL = {
        "width": 68.0,
        "height": 105.0,
        "name": "Women's Professional",
        "description": "Professional women's football field dimensions",
    }

    # American soccer variations
    MLS = {
        "width": 70.0,
        "height": 110.0,
        "name": "MLS",
        "description": "Major League Soccer standard dimensions",
    }

    # School/College fields
    HIGH_SCHOOL = {
        "width": 55.0,
        "height": 100.0,
        "name": "High School",
        "description": "Typical high school soccer field dimensions",
    }

    COLLEGE = {
        "width": 68.0,
        "height": 105.0,
        "name": "College",
        "description": "College soccer field dimensions",
    }

    @classmethod
    def get_all_presets(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available field dimension presets."""
        presets = {}
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and not callable(getattr(cls, attr_name)):
                attr_value = getattr(cls, attr_name)
                if (
                    isinstance(attr_value, dict)
                    and "width" in attr_value
                    and "height" in attr_value
                ):
                    presets[attr_name] = attr_value
        return presets

    @classmethod
    def get_dimensions(cls, preset_name: str) -> Tuple[float, float]:
        """
        Get width and height for a preset.

        Args:
            preset_name: Name of the preset (e.g., 'FIFA_STANDARD')

        Returns:
            Tuple of (width, height) in meters

        Raises:
            ValueError: If preset not found
        """
        preset = getattr(cls, preset_name, None)
        if preset is None or not isinstance(preset, dict):
            available = [
                name
                for name in dir(cls)
                if not name.startswith("_") and not callable(getattr(cls, name))
            ]
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        return preset["width"], preset["height"]

    @classmethod
    def list_presets(cls) -> None:
        """Print all available presets with their descriptions."""
        presets = cls.get_all_presets()
        print("Available Field Dimension Presets:")
        print("=" * 50)
        for name, info in presets.items():
            print(
                f"{name:20} | {info['width']:5.1f}m x {info['height']:6.1f}m | {info['description']}"
            )


# Quick access functions
def get_fifa_standard() -> Tuple[float, float]:
    """Get FIFA standard field dimensions."""
    return (
        FieldDimensions.FIFA_STANDARD["width"],
        FieldDimensions.FIFA_STANDARD["height"],
    )


def get_youth_dimensions(age_group: str) -> Tuple[float, float]:
    """
    Get youth field dimensions by age group.

    Args:
        age_group: 'U12', 'U14', or 'U16'

    Returns:
        Tuple of (width, height) in meters
    """
    preset_map = {"U12": "YOUTH_U12", "U14": "YOUTH_U14", "U16": "YOUTH_U16"}

    if age_group not in preset_map:
        raise ValueError(
            f"Unknown age group '{age_group}'. Available: {list(preset_map.keys())}"
        )

    return FieldDimensions.get_dimensions(preset_map[age_group])


def get_league_dimensions(league: str) -> Tuple[float, float]:
    """
    Get field dimensions for a specific league.

    Args:
        league: League name ('premier_league', 'la_liga', 'bundesliga', 'serie_a', 'mls', etc.)

    Returns:
        Tuple of (width, height) in meters
    """
    league_map = {
        "premier_league": "PREMIER_LEAGUE",
        "la_liga": "LA_LIGA",
        "bundesliga": "BUNDESLIGA",
        "serie_a": "SERIE_A",
        "mls": "MLS",
        "uefa_champions": "UEFA_CHAMPIONS",
    }

    league_key = league.lower()
    if league_key not in league_map:
        raise ValueError(
            f"Unknown league '{league}'. Available: {list(league_map.keys())}"
        )

    return FieldDimensions.get_dimensions(league_map[league_key])
