"""
Unit Tests for Football Analysis Core Models

Tests all core data models including Player, Ball, Frame, Video, Field, etc.
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock

# Import core models
from football_ai.core_models.player import Player
from football_ai.core_models.ball import Ball
from football_ai.core_models.goalkeeper import Goalkeeper
from football_ai.core_models.referee import Referee
from football_ai.core_models.field import Field
from football_ai.core_models.frame import Frame, CameraMotion, ProcessingStatus
from football_ai.core_models.video import (
    Video,
    VideoMetadata,
    MatchContext,
    ProcessingConfig,
)
from football_ai.core_models.game import Game, Team, MatchType, MatchStatus
from football_ai.core_models.interfaces import Processor


class TestPlayer(unittest.TestCase):
    """Test Player model."""

    def setUp(self):
        """Set up test fixtures."""
        self.player = Player(
            track_id=1,
            player_id="P001",
            jersey_number=10,
            team_id=1,
            pixel_position=(100.0, 200.0),
            field_position=(25.0, 35.0),
            speed=5.5,
            direction=45.0,
            track_confidence=0.90,
            is_active=True,
        )

    def test_player_creation(self):
        """Test basic player creation."""
        self.assertEqual(self.player.track_id, 1)
        self.assertEqual(self.player.player_id, "P001")
        self.assertEqual(self.player.jersey_number, 10)
        self.assertEqual(self.player.team_id, 1)
        self.assertEqual(self.player.pixel_position, (100.0, 200.0))
        self.assertEqual(self.player.field_position, (25.0, 35.0))
        self.assertEqual(self.player.speed, 5.5)
        self.assertEqual(self.player.direction, 45.0)
        self.assertTrue(self.player.is_active)

    def test_player_with_minimal_data(self):
        """Test player creation with minimal required data."""
        minimal_player = Player(track_id=99)
        self.assertEqual(minimal_player.track_id, 99)
        self.assertIsNone(minimal_player.player_id)
        self.assertIsNone(minimal_player.team_id)
        self.assertTrue(minimal_player.is_active)

    def test_player_position_updates(self):
        """Test updating player positions."""
        self.player.pixel_position = (150.0, 250.0)
        self.player.field_position = (30.0, 40.0)

        self.assertEqual(self.player.pixel_position, (150.0, 250.0))
        self.assertEqual(self.player.field_position, (30.0, 40.0))

    def test_player_performance_stats(self):
        """Test player performance statistics."""
        self.player.total_distance = 1500.0
        self.player.max_speed = 12.5
        self.player.avg_speed = 6.2
        self.player.sprint_count = 3

        self.assertEqual(self.player.total_distance, 1500.0)
        self.assertEqual(self.player.max_speed, 12.5)
        self.assertEqual(self.player.avg_speed, 6.2)
        self.assertEqual(self.player.sprint_count, 3)

    def test_update_position(self):
        """Test updating player position with history."""
        self.player.update_position((150.0, 250.0), (30.0, 40.0))

        self.assertEqual(self.player.pixel_position, (150.0, 250.0))
        self.assertEqual(self.player.field_position, (30.0, 40.0))
        self.assertIn((150.0, 250.0), self.player.position_history)
        self.assertIn((30.0, 40.0), self.player.field_position_history)

    def test_update_position_pixel_only(self):
        """Test updating only pixel position."""
        self.player.update_position((175.0, 275.0))

        self.assertEqual(self.player.pixel_position, (175.0, 275.0))
        self.assertIn((175.0, 275.0), self.player.position_history)

    def test_calculate_distance_traveled(self):
        """Test distance calculation from position history."""
        # Test with no history
        self.assertEqual(self.player.calculate_distance_traveled(), 0.0)

        # Add positions to history
        self.player.position_history = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]

        # Distance should be 5.0 + 5.0 = 10.0
        expected_distance = 5.0 + 5.0  # sqrt(3²+4²) + sqrt(3²+4²)
        self.assertAlmostEqual(
            self.player.calculate_distance_traveled(), expected_distance, places=2
        )

    def test_is_in_possession(self):
        """Test ball possession check."""
        self.assertFalse(self.player.is_in_possession())

        self.player.has_ball = True
        self.assertTrue(self.player.is_in_possession())

    def test_get_current_speed(self):
        """Test getting current speed."""
        self.assertEqual(self.player.get_current_speed(), 5.5)

        self.player.speed = None
        self.assertIsNone(self.player.get_current_speed())

    def test_get_position_at_frame(self):
        """Test getting position at specific frame."""
        self.player.position_history = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]

        self.assertEqual(self.player.get_position_at_frame(1), (10.0, 10.0))
        self.assertIsNone(self.player.get_position_at_frame(5))  # Out of bounds
        self.assertIsNone(self.player.get_position_at_frame(-1))  # Negative index

    def test_get_field_position_at_frame(self):
        """Test getting field position at specific frame."""
        self.player.field_position_history = [(0.0, 0.0), (25.0, 35.0), (50.0, 70.0)]

        self.assertEqual(self.player.get_field_position_at_frame(1), (25.0, 35.0))
        self.assertIsNone(self.player.get_field_position_at_frame(5))  # Out of bounds

    def test_calculate_average_speed(self):
        """Test average speed calculation."""
        # Test with no data
        self.assertEqual(self.player.calculate_average_speed(), 0.0)

        # Test with valid data
        self.player.total_distance = 100.0
        self.player.track_age = 10
        self.assertEqual(self.player.calculate_average_speed(), 10.0)

    def test_update_statistics(self):
        """Test updating player statistics."""
        self.player.update_statistics(10.5, 8.2, 1625090000.0)

        self.assertEqual(self.player.total_distance, 10.5)
        self.assertEqual(self.player.speed, 8.2)
        self.assertEqual(self.player.max_speed, 8.2)
        self.assertEqual(self.player.frame_timestamp, 1625090000.0)

        # Test updating with higher speed
        self.player.update_statistics(5.0, 12.5)
        self.assertEqual(self.player.total_distance, 15.5)
        self.assertEqual(self.player.max_speed, 12.5)

    def test_is_goalkeeper(self):
        """Test goalkeeper identification."""
        self.assertFalse(self.player.is_goalkeeper())

        self.player.position_role = "goalkeeper"
        self.assertTrue(self.player.is_goalkeeper())

        self.player.position_role = "defender"
        self.player.formation_position = "GK"
        self.assertTrue(self.player.is_goalkeeper())

    def test_get_team_info(self):
        """Test getting team information."""
        self.player.team_assignment_confidence = 0.85

        team_info = self.player.get_team_info()
        expected = {
            "team_id": 1,
            "team_assignment_confidence": 0.85,
            "jersey_number": 10,
        }
        self.assertEqual(team_info, expected)

    def test_get_tracking_info(self):
        """Test getting tracking information."""
        tracking_info = self.player.get_tracking_info()
        expected = {
            "track_id": 1,
            "track_confidence": 0.90,
            "track_age": None,
            "last_seen_frame": None,
            "is_active": True,
        }
        self.assertEqual(tracking_info, expected)

    def test_add_position_data(self):
        """Test adding position data."""
        self.player.add_position_data(
            timestamp=1625090000.0,
            x=100.0,
            y=200.0,
            velocity_x=2.0,
            velocity_y=3.0,
            speed=5.0,
        )

        self.assertEqual(self.player.pixel_position, (100.0, 200.0))
        self.assertEqual(self.player.speed, 5.0)
        self.assertEqual(self.player.max_speed, 5.0)
        self.assertEqual(self.player.frame_timestamp, 1625090000.0)
        self.assertIn((100.0, 200.0), self.player.position_history)

    def test_add_position_data_calculated_speed(self):
        """Test adding position data with calculated speed."""
        self.player.add_position_data(
            timestamp=1625090000.0,
            x=100.0,
            y=200.0,
            velocity_x=3.0,
            velocity_y=4.0,
        )

        # Speed should be calculated as sqrt(3²+4²) = 5.0
        self.assertIsNotNone(self.player.speed)
        if self.player.speed is not None:
            self.assertAlmostEqual(self.player.speed, 5.0, places=2)

    def test_player_tactical_info(self):
        """Test tactical information."""
        self.player.position_role = "midfielder"
        self.player.tactical_role = "box-to-box"
        self.player.formation_position = "CM"

        self.assertEqual(self.player.position_role, "midfielder")
        self.assertEqual(self.player.tactical_role, "box-to-box")
        self.assertEqual(self.player.formation_position, "CM")


class TestBall(unittest.TestCase):
    """Test Ball model."""

    def setUp(self):
        """Set up test fixtures."""
        self.ball = Ball(
            track_id=100,
            pixel_position=(200.0, 300.0),
            field_position=(50.0, 25.0),
            speed=15.0,
            direction=90.0,
            is_active=True,
            is_visible=True,
        )

    def test_ball_creation(self):
        """Test basic ball creation."""
        self.assertEqual(self.ball.track_id, 100)
        self.assertEqual(self.ball.pixel_position, (200.0, 300.0))
        self.assertEqual(self.ball.field_position, (50.0, 25.0))
        self.assertEqual(self.ball.speed, 15.0)
        self.assertEqual(self.ball.direction, 90.0)
        self.assertTrue(self.ball.is_active)
        self.assertTrue(self.ball.is_visible)

    def test_ball_with_minimal_data(self):
        """Test ball creation with minimal required data."""
        minimal_ball = Ball(track_id=99)
        self.assertEqual(minimal_ball.track_id, 99)
        self.assertIsNone(minimal_ball.ball_id)
        self.assertTrue(minimal_ball.is_active)
        self.assertTrue(minimal_ball.is_visible)

    def test_ball_position_history(self):
        """Test ball position history tracking."""
        self.assertEqual(len(self.ball.position_history), 0)
        self.assertEqual(len(self.ball.field_position_history), 0)
        self.assertEqual(len(self.ball.speed_history), 0)

        # Add some history
        self.ball.position_history.append((200.0, 300.0))
        self.ball.field_position_history.append((50.0, 25.0))
        self.ball.speed_history.append(15.0)

        self.assertEqual(len(self.ball.position_history), 1)
        self.assertEqual(len(self.ball.field_position_history), 1)
        self.assertEqual(len(self.ball.speed_history), 1)

    def test_ball_velocity_vector(self):
        """Test ball velocity vector."""
        self.ball.velocity_vector = (10.0, 5.0)
        self.assertEqual(self.ball.velocity_vector, (10.0, 5.0))

    def test_ball_tracking_data(self):
        """Test ball tracking data."""
        self.ball.track_confidence = 0.88
        self.ball.track_age = 25
        self.ball.last_seen_frame = 100

        self.assertEqual(self.ball.track_confidence, 0.88)
        self.assertEqual(self.ball.track_age, 25)
        self.assertEqual(self.ball.last_seen_frame, 100)

    def test_ball_possession_data(self):
        """Test ball possession tracking."""
        self.ball.controlling_player_id = 5
        self.ball.last_touch_frame = 50
        self.ball.possession_confidence = 0.75
        self.ball.possession_team_id = 1

        self.assertEqual(self.ball.controlling_player_id, 5)
        self.assertEqual(self.ball.last_touch_frame, 50)
        self.assertEqual(self.ball.possession_confidence, 0.75)
        self.assertEqual(self.ball.possession_team_id, 1)

    def test_ball_movement_calculation(self):
        """Test ball movement calculations."""
        # Setup position history
        self.ball.position_history = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]

        # Calculate distance (should be 5.0 + 5.0 = 10.0)
        total_distance = 0.0
        for i in range(1, len(self.ball.position_history)):
            prev_pos = self.ball.position_history[i - 1]
            curr_pos = self.ball.position_history[i]
            distance = (
                (curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2
            ) ** 0.5
            total_distance += distance

        self.assertAlmostEqual(total_distance, 10.0, places=2)

    def test_ball_speed_tracking(self):
        """Test ball speed tracking over time."""
        speeds = [10.0, 15.0, 20.0, 12.0, 8.0]
        self.ball.speed_history = speeds

        self.assertEqual(len(self.ball.speed_history), 5)
        self.assertEqual(max(self.ball.speed_history), 20.0)
        self.assertEqual(min(self.ball.speed_history), 8.0)
        self.assertAlmostEqual(sum(speeds) / len(speeds), 13.0, places=2)

    def test_ball_visibility_states(self):
        """Test ball visibility states."""
        self.assertTrue(self.ball.is_visible)

        # Test occlusion
        self.ball.is_visible = False
        self.assertFalse(self.ball.is_visible)

        # Test with active but not visible
        self.ball.is_active = True
        self.ball.is_visible = False
        self.assertTrue(self.ball.is_active)
        self.assertFalse(self.ball.is_visible)

    def test_ball_field_position_conversion(self):
        """Test ball field position conversion."""
        self.ball.pixel_position = (320, 240)
        self.ball.field_position = (52.5, 34.0)  # Center of field

        self.assertEqual(self.ball.pixel_position, (320, 240))
        self.assertEqual(self.ball.field_position, (52.5, 34.0))

    def test_ball_state_analysis(self):
        """Test ball state analysis features."""
        self.ball.ball_state = "controlled"
        self.ball.is_in_play = True
        self.ball.is_airborne = False
        self.ball.estimated_height = None

        self.assertEqual(self.ball.ball_state, "controlled")
        self.assertTrue(self.ball.is_in_play)
        self.assertFalse(self.ball.is_airborne)
        self.assertIsNone(self.ball.estimated_height)

        # Test airborne state
        self.ball.is_airborne = True
        self.ball.estimated_height = 2.5
        self.assertTrue(self.ball.is_airborne)
        self.assertEqual(self.ball.estimated_height, 2.5)

    def test_ball_field_zone_tracking(self):
        """Test ball field zone tracking."""
        self.ball.current_field_zone = "middle_third"
        self.ball.time_in_zones = {
            "defensive_third": 120.0,
            "middle_third": 180.0,
            "attacking_third": 100.0,
        }

        self.assertEqual(self.ball.current_field_zone, "middle_third")
        self.assertEqual(self.ball.time_in_zones["middle_third"], 180.0)
        self.assertEqual(sum(self.ball.time_in_zones.values()), 400.0)

    def test_ball_events_tracking(self):
        """Test ball events tracking."""
        touch_event = {
            "frame": 100,
            "player_id": 5,
            "position": (50.0, 25.0),
            "type": "pass",
        }

        self.ball.touches.append(touch_event)
        self.assertEqual(len(self.ball.touches), 1)
        self.assertEqual(self.ball.touches[0]["player_id"], 5)

    def test_ball_update_position(self):
        """Test ball position update method."""
        initial_history_length = len(self.ball.position_history)

        self.ball.update_position((250.0, 350.0), (55.0, 30.0))

        self.assertEqual(self.ball.pixel_position, (250.0, 350.0))
        self.assertEqual(self.ball.field_position, (55.0, 30.0))
        self.assertEqual(len(self.ball.position_history), initial_history_length + 1)
        self.assertEqual(
            len(self.ball.field_position_history), 1
        )  # Assuming it was empty


class TestGoalkeeper(unittest.TestCase):
    """Test Goalkeeper model."""

    def setUp(self):
        """Set up test fixtures."""
        self.goalkeeper = Goalkeeper(
            track_id=1,
            player_id="GK001",
            jersey_number=1,
            team_id=1,
            pixel_position=(50.0, 240.0),
            field_position=(5.0, 34.0),
            speed=3.0,
            is_active=True,
        )

    def test_goalkeeper_creation(self):
        """Test basic goalkeeper creation."""
        self.assertEqual(self.goalkeeper.track_id, 1)
        self.assertEqual(self.goalkeeper.player_id, "GK001")
        self.assertEqual(self.goalkeeper.jersey_number, 1)
        self.assertEqual(self.goalkeeper.team_id, 1)
        self.assertTrue(self.goalkeeper.is_active)

    def test_goalkeeper_specific_attributes(self):
        """Test goalkeeper-specific attributes."""
        # Set goalkeeper-specific attributes that exist in the class
        self.goalkeeper.saves_count = 3
        self.goalkeeper.goals_conceded = 1
        self.goalkeeper.clean_sheet = False
        self.goalkeeper.distribution_accuracy = 0.85
        self.goalkeeper.shots_faced = 5

        self.assertEqual(self.goalkeeper.saves_count, 3)
        self.assertEqual(self.goalkeeper.goals_conceded, 1)
        self.assertFalse(self.goalkeeper.clean_sheet)
        self.assertEqual(self.goalkeeper.distribution_accuracy, 0.85)
        self.assertEqual(self.goalkeeper.shots_faced, 5)

    def test_goalkeeper_area_coverage(self):
        """Test goalkeeper area coverage tracking."""
        # Test goal area positioning using correct attributes
        self.goalkeeper.is_in_penalty_area = True
        self.goalkeeper.is_in_goal_area = True
        self.goalkeeper.penalty_area_position = (10.0, 34.0)
        self.goalkeeper.furthest_from_goal = 15.5

        self.assertTrue(self.goalkeeper.is_in_penalty_area)
        self.assertTrue(self.goalkeeper.is_in_goal_area)
        self.assertEqual(self.goalkeeper.penalty_area_position, (10.0, 34.0))
        self.assertEqual(self.goalkeeper.furthest_from_goal, 15.5)


class TestReferee(unittest.TestCase):
    """Test Referee model."""

    def setUp(self):
        """Set up test fixtures."""
        self.referee = Referee(
            track_id=200,
            referee_id="REF001",
            referee_type="main",
            pixel_position=(320.0, 240.0),
            field_position=(52.5, 34.0),
            speed=2.5,
            is_active=True,
        )

    def test_referee_creation(self):
        """Test basic referee creation."""
        self.assertEqual(self.referee.track_id, 200)
        self.assertEqual(self.referee.referee_id, "REF001")
        self.assertEqual(self.referee.referee_type, "main")
        self.assertTrue(self.referee.is_active)

    def test_referee_types(self):
        """Test different referee types."""
        assistant_ref = Referee(track_id=201, referee_type="assistant")
        fourth_official = Referee(track_id=202, referee_type="fourth_official")

        self.assertEqual(assistant_ref.referee_type, "assistant")
        self.assertEqual(fourth_official.referee_type, "fourth_official")

    def test_referee_positioning(self):
        """Test referee positioning tracking."""
        self.referee.pixel_position = (400.0, 300.0)
        self.referee.field_position = (60.0, 40.0)

        self.assertEqual(self.referee.pixel_position, (400.0, 300.0))
        self.assertEqual(self.referee.field_position, (60.0, 40.0))

    def test_referee_movement_tracking(self):
        """Test referee movement tracking."""
        self.referee.position_history = [(320.0, 240.0), (330.0, 250.0), (340.0, 260.0)]

        # Calculate total distance
        total_distance = 0.0
        for i in range(1, len(self.referee.position_history)):
            prev_pos = self.referee.position_history[i - 1]
            curr_pos = self.referee.position_history[i]
            distance = (
                (curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2
            ) ** 0.5
            total_distance += distance

        expected_distance = 2 * ((10**2 + 10**2) ** 0.5)  # Two movements of ~14.14 each
        self.assertAlmostEqual(total_distance, expected_distance, places=2)


class TestField(unittest.TestCase):
    """Test Field model - Enhanced."""

    def setUp(self):
        """Set up test fixtures."""
        self.field = Field(
            field_id="field_001",
            field_name="Stadium A",
            length=105.0,
            width=68.0,
            field_corners=[(0, 0), (640, 0), (640, 480), (0, 480)],
        )

    def test_field_creation(self):
        """Test basic field creation."""
        self.assertEqual(self.field.field_id, "field_001")
        self.assertEqual(self.field.field_name, "Stadium A")
        self.assertEqual(self.field.length, 105.0)
        self.assertEqual(self.field.width, 68.0)

    def test_field_standard_dimensions(self):
        """Test field with standard dimensions."""
        self.assertEqual(self.field.length, 105.0)
        self.assertEqual(self.field.width, 68.0)
        self.assertEqual(self.field.goal_width, 7.32)
        self.assertEqual(self.field.goal_height, 2.44)
        self.assertEqual(self.field.penalty_area_length, 16.5)
        self.assertEqual(self.field.center_circle_radius, 9.15)

    def test_field_corners(self):
        """Test field corner coordinates."""
        expected_corners = [(0, 0), (640, 0), (640, 480), (0, 480)]
        self.assertEqual(self.field.field_corners, expected_corners)

    def test_field_goal_coordinates(self):
        """Test field goal coordinates."""
        self.field.left_goal_posts = [(10, 200), (10, 280), (0, 200), (0, 280)]
        self.field.right_goal_posts = [(630, 200), (630, 280), (640, 200), (640, 280)]

        self.assertEqual(len(self.field.left_goal_posts), 4)
        self.assertEqual(len(self.field.right_goal_posts), 4)

    def test_field_area_definitions(self):
        """Test field area definitions."""
        # Test penalty area dimensions
        self.assertEqual(self.field.penalty_area_width, 40.32)
        self.assertEqual(self.field.goal_area_length, 5.5)
        self.assertEqual(self.field.goal_area_width, 18.32)

    def test_field_transformation_methods(self):
        """Test field transformation methods if they exist."""
        # These methods might be added to Field class later
        pass
