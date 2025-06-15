#!/usr/bin/env python3
"""
Field Preset Demonstration

This script demonstrates how to use the field dimension presets and
convenience methods for creating coordinate transformers for different
types of soccer fields.
"""

from football_ai.constants import FieldDimensions
from football_ai.transformation.coordinate_transformer import (
    PerspectiveCoordinateTransformer,
)


def main():
    print("=" * 60)
    print("Football Field Dimension Presets Demonstration")
    print("=" * 60)
    print()

    # Show all available presets
    print("1. Available Field Presets:")
    print("-" * 40)
    FieldDimensions.list_presets()
    print()

    # Demonstrate different ways to create transformers
    print("2. Creating Transformers with Different Methods:")
    print("-" * 50)

    # Method 1: Convenience class methods
    print("a) Using convenience class methods:")

    fifa_transformer = PerspectiveCoordinateTransformer.for_fifa_standard()
    print(
        f"   FIFA Standard: {fifa_transformer.field_width}m x {fifa_transformer.field_height}m"
    )

    youth_transformer = PerspectiveCoordinateTransformer.for_youth("U12")
    print(
        f"   Youth U12: {youth_transformer.field_width}m x {youth_transformer.field_height}m"
    )

    mls_transformer = PerspectiveCoordinateTransformer.for_league("mls")
    print(f"   MLS: {mls_transformer.field_width}m x {mls_transformer.field_height}m")

    small_transformer = PerspectiveCoordinateTransformer.for_small_sided("5-a-side")
    print(
        f"   5-a-side: {small_transformer.field_width}m x {small_transformer.field_height}m"
    )
    print()

    # Method 2: Direct preset usage
    print("b) Using field_preset parameter:")
    preset_transformer = PerspectiveCoordinateTransformer(field_preset="BUNDESLIGA")
    print(
        f"   Bundesliga: {preset_transformer.field_width}m x {preset_transformer.field_height}m"
    )

    uefa_transformer = PerspectiveCoordinateTransformer(field_preset="UEFA_CHAMPIONS")
    print(
        f"   UEFA Champions: {uefa_transformer.field_width}m x {uefa_transformer.field_height}m"
    )
    print()

    # Method 3: Manual dimensions (overrides preset)
    print("c) Using manual dimensions (overrides preset):")
    custom_transformer = PerspectiveCoordinateTransformer(
        field_width=75.0, field_height=110.0
    )
    print(
        f"   Custom field: {custom_transformer.field_width}m x {custom_transformer.field_height}m"
    )
    print()

    # Show detailed information
    print("3. Detailed Transformer Information:")
    print("-" * 40)
    info = fifa_transformer.get_calibration_info()
    print("FIFA Standard transformer details:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # Demonstrate field zone calculations
    print("4. Field Zone Calculations (different field sizes):")
    print("-" * 50)

    transformers = [
        ("FIFA Standard", fifa_transformer),
        ("Youth U12", youth_transformer),
        ("MLS", mls_transformer),
        ("5-a-side", small_transformer),
    ]

    for name, transformer in transformers:
        # Test center of field
        center_x = transformer.field_width / 2
        center_y = transformer.field_height / 2
        zone = transformer.get_field_zone((center_x, center_y))

        # Test corner
        corner_x = transformer.field_width * 0.1
        corner_y = transformer.field_height * 0.1
        corner_zone = transformer.get_field_zone((corner_x, corner_y))

        print(
            f"  {name:15} | Center ({center_x:4.1f}, {center_y:5.1f}): {zone:15} | Corner: {corner_zone}"
        )
    print()

    # Show distance calculations
    print("5. Distance Calculations (same relative positions, different field sizes):")
    print("-" * 70)

    # Points at 25% and 75% of field length
    for name, transformer in transformers:
        point1 = (transformer.field_width * 0.25, transformer.field_height * 0.25)
        point2 = (transformer.field_width * 0.75, transformer.field_height * 0.75)
        distance = transformer.calculate_distance(point1, point2)

        print(f"  {name:15} | Quarter to three-quarter diagonal: {distance:.1f}m")
    print()

    # Show how to handle different leagues
    print("6. League-Specific Examples:")
    print("-" * 30)

    leagues = ["premier_league", "la_liga", "bundesliga", "serie_a", "mls"]
    for league in leagues:
        try:
            league_transformer = PerspectiveCoordinateTransformer.for_league(league)
            print(
                f"  {league.replace('_', ' ').title():15}: {league_transformer.field_width}m x {league_transformer.field_height}m"
            )
        except ValueError as e:
            print(f"  {league}: Error - {e}")
    print()

    print("7. Usage in Pipeline:")
    print("-" * 20)
    print("To use in the football analysis pipeline:")
    print()
    print("# For FIFA standard:")
    print("transformer = PerspectiveCoordinateTransformer.for_fifa_standard()")
    print()
    print("# For specific league:")
    print("transformer = PerspectiveCoordinateTransformer.for_league('mls')")
    print()
    print("# For youth soccer:")
    print("transformer = PerspectiveCoordinateTransformer.for_youth('U14')")
    print()
    print("# Custom field size:")
    print(
        "transformer = PerspectiveCoordinateTransformer(field_width=70.0, field_height=100.0)"
    )
    print()
    print("# Then set field corners and use as normal:")
    print("transformer.set_field_corners(detected_corners)")
    print("field_coord = transformer.transform_point(pixel_coord)")
    print()

    print("=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
