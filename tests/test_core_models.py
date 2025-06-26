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
from unittest.mock import Mock, patch, MagicMock

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
            detection_confidence=0.95,
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
            detection_confidence=0.85,
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


class TestField(unittest.TestCase):
    """Test Field model."""

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
        self.assertEqual(self.field.field_type, "football")

    def test_field_standard_dimensions(self):
        """Test field with standard dimensions."""
        standard_field = Field()
        self.assertEqual(standard_field.length, 105.0)
        self.assertEqual(standard_field.width, 68.0)
        self.assertEqual(standard_field.goal_width, 7.32)
        self.assertEqual(standard_field.goal_height, 2.44)

    def test_field_corners(self):
        """Test field corner coordinates."""
        expected_corners = [(0, 0), (640, 0), (640, 480), (0, 480)]
        self.assertEqual(self.field.field_corners, expected_corners)


class TestFrame(unittest.TestCase):
    """Test Frame model."""

    def setUp(self):
        """Set up test fixtures."""
        self.frame = Frame(
            frame_number=1,
            timestamp=0.033,
            frame_id="frame_001",
            raw_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            frame_shape=(480, 640, 3),
        )

    def test_frame_creation(self):
        """Test basic frame creation."""
        self.assertEqual(self.frame.frame_number, 1)
        self.assertEqual(self.frame.timestamp, 0.033)
        self.assertEqual(self.frame.frame_id, "frame_001")
        self.assertEqual(self.frame.frame_shape, (480, 640, 3))
        self.assertIsNotNone(self.frame.raw_frame)

    def test_frame_processing_status(self):
        """Test frame processing status."""
        self.assertFalse(self.frame.processing_status.detection_processed)
        self.assertFalse(self.frame.processing_status.tracking_processed)
        self.assertFalse(self.frame.processing_status.is_complete())

        # Update processing status
        self.frame.processing_status.detection_processed = True
        self.frame.processing_status.tracking_processed = True
        self.assertTrue(self.frame.processing_status.detection_processed)
        self.assertTrue(self.frame.processing_status.tracking_processed)

    def test_frame_camera_motion(self):
        """Test frame camera motion data."""
        self.frame.camera_motion.x_offset = 5.0
        self.frame.camera_motion.y_offset = -2.0
        self.frame.camera_motion.rotation = 1.5
        self.frame.camera_motion.zoom = 1.1

        self.assertEqual(self.frame.camera_motion.x_offset, 5.0)
        self.assertEqual(self.frame.camera_motion.y_offset, -2.0)
        self.assertEqual(self.frame.camera_motion.rotation, 1.5)
        self.assertEqual(self.frame.camera_motion.zoom, 1.1)
        self.assertTrue(bool(self.frame.camera_motion))

    def test_frame_objects(self):
        """Test frame object collections."""
        # Add a player
        player = Player(track_id=1, team_id=1)
        self.frame.players[1] = player

        # Add a ball
        ball = Ball(track_id=100)
        self.frame.ball = ball

        self.assertEqual(len(self.frame.players), 1)
        self.assertIsNotNone(self.frame.ball)
        self.assertEqual(self.frame.players[1].track_id, 1)
        self.assertEqual(self.frame.ball.track_id, 100)


class TestVideo(unittest.TestCase):
    """Test Video model."""

    def setUp(self):
        """Set up test fixtures."""
        self.video_metadata = VideoMetadata(
            file_path="/test/video.mp4",
            frame_rate=30.0,
            resolution=(1920, 1080),
            duration=60.0,
            total_frames=1800,
        )

        self.match_context = MatchContext(
            match_id="match_001",
            home_team="Team A",
            away_team="Team B",
            final_score=(2, 1),
        )

        self.video = Video(
            video_id="video_001",
            video_path="/test/video.mp4",
            metadata=self.video_metadata,
            match_context=self.match_context,
        )

    def test_video_creation(self):
        """Test basic video creation."""
        self.assertEqual(self.video.video_id, "video_001")
        self.assertEqual(self.video.video_path, "/test/video.mp4")
        self.assertEqual(self.video.metadata.frame_rate, 30.0)
        self.assertEqual(self.video.match_context.home_team, "Team A")

    def test_video_frame_management(self):
        """Test video frame management."""
        # Initially no frames
        self.assertEqual(len(self.video.frames), 0)

        # Add frames
        frame1 = Frame(frame_number=1, timestamp=0.033)
        frame2 = Frame(frame_number=2, timestamp=0.066)

        self.video.add_frame(frame1)
        self.video.add_frame(frame2)

        self.assertEqual(len(self.video.frames), 2)
        self.assertEqual(self.video.frames[0].frame_number, 1)
        self.assertEqual(self.video.frames[1].frame_number, 2)

    def test_video_frame_retrieval(self):
        """Test video frame retrieval methods."""
        frame1 = Frame(frame_number=1, timestamp=0.033)
        frame2 = Frame(frame_number=2, timestamp=0.066)

        self.video.add_frame(frame1)
        self.video.add_frame(frame2)

        # Get frame by number
        retrieved_frame = self.video.get_frame(1)
        self.assertIsNotNone(retrieved_frame)
        if retrieved_frame:
            self.assertEqual(retrieved_frame.frame_number, 1)

        # Get frame by timestamp
        retrieved_frame = self.video.get_frame_by_timestamp(0.033)
        self.assertIsNotNone(retrieved_frame)
        if retrieved_frame:
            self.assertEqual(retrieved_frame.timestamp, 0.033)

    def test_video_frame_removal(self):
        """Test video frame removal."""
        frame1 = Frame(frame_number=1, timestamp=0.033)
        self.video.add_frame(frame1)

        self.assertEqual(len(self.video.frames), 1)

        removed_frame = self.video.remove_frame(1)
        self.assertIsNotNone(removed_frame)
        if removed_frame:
            self.assertEqual(removed_frame.frame_number, 1)
        self.assertEqual(len(self.video.frames), 0)

    def test_video_field_management(self):
        """Test video field management."""
        field = Field(field_id="field_001")
        self.video.set_global_field(field)

        self.assertIsNotNone(self.video.global_field)
        if self.video.global_field:
            self.assertEqual(self.video.global_field.field_id, "field_001")

    def test_video_duplicate_frame_error(self):
        """Test that adding duplicate frame numbers raises error."""
        frame1 = Frame(frame_number=1, timestamp=0.033)
        frame2 = Frame(frame_number=1, timestamp=0.066)  # Same frame number

        self.video.add_frame(frame1)

        with self.assertRaises(ValueError):
            self.video.add_frame(frame2)


class TestGame(unittest.TestCase):
    """Test Game model."""

    def setUp(self):
        """Set up test fixtures."""
        from datetime import date

        self.home_team = Team(team_id="team_a", name="Team A", short_name="TA")
        self.away_team = Team(team_id="team_b", name="Team B", short_name="TB")

        self.game = Game(
            game_id="game_001",
            match_type=MatchType.LEAGUE,
            status=MatchStatus.FINISHED,
            home_team=self.home_team,
            away_team=self.away_team,
            match_date=date.today(),
        )

    def test_game_creation(self):
        """Test basic game creation."""
        self.assertEqual(self.game.game_id, "game_001")
        self.assertEqual(self.game.match_type, MatchType.LEAGUE)
        self.assertEqual(self.game.status, MatchStatus.FINISHED)
        self.assertEqual(self.game.home_team.name, "Team A")
        self.assertEqual(self.game.away_team.name, "Team B")

    def test_team_creation(self):
        """Test team creation."""
        self.assertEqual(self.home_team.team_id, "team_a")
        self.assertEqual(self.home_team.name, "Team A")
        self.assertEqual(self.home_team.short_name, "TA")


class TestGoalkeeper(unittest.TestCase):
    """Test Goalkeeper model."""

    def setUp(self):
        """Set up test fixtures."""
        self.goalkeeper = Goalkeeper(
            track_id=50, player_id="GK001", jersey_number=1, team_id=1, is_active=True
        )

    def test_goalkeeper_creation(self):
        """Test basic goalkeeper creation."""
        self.assertEqual(self.goalkeeper.track_id, 50)
        self.assertEqual(self.goalkeeper.player_id, "GK001")
        self.assertEqual(self.goalkeeper.jersey_number, 1)
        self.assertEqual(self.goalkeeper.team_id, 1)
        self.assertTrue(self.goalkeeper.is_active)


class TestReferee(unittest.TestCase):
    """Test Referee model."""

    def setUp(self):
        """Set up test fixtures."""
        self.referee = Referee(track_id=200, referee_id="REF001", is_active=True)

    def test_referee_creation(self):
        """Test basic referee creation."""
        self.assertEqual(self.referee.track_id, 200)
        self.assertEqual(self.referee.referee_id, "REF001")
        self.assertTrue(self.referee.is_active)


class TestProcessorInterface(unittest.TestCase):
    """Test Processor interface."""

    def test_processor_interface(self):
        """Test that Processor is an abstract base class."""
        # Should not be able to instantiate abstract class directly
        # This is expected behavior for abstract base classes
        self.assertTrue(hasattr(Processor, "process"))

    def test_processor_implementation(self):
        """Test implementing Processor interface."""

        class TestProcessor(Processor):
            def process(self, data: Video) -> Video:
                return data

        # Should be able to instantiate concrete implementation
        processor = TestProcessor()
        self.assertIsInstance(processor, Processor)

        # Test process method
        video = Video(video_id="test", video_path="test.mp4")
        result = processor.process(video)
        self.assertEqual(result, video)


class TestCameraMotion(unittest.TestCase):
    """Test CameraMotion model."""

    def test_camera_motion_creation(self):
        """Test camera motion creation."""
        motion = CameraMotion(x_offset=5.0, y_offset=-2.0, rotation=1.5, zoom=1.1)

        self.assertEqual(motion.x_offset, 5.0)
        self.assertEqual(motion.y_offset, -2.0)
        self.assertEqual(motion.rotation, 1.5)
        self.assertEqual(motion.zoom, 1.1)
        self.assertTrue(bool(motion))

    def test_camera_motion_empty(self):
        """Test empty camera motion."""
        motion = CameraMotion()
        self.assertFalse(bool(motion))


class TestProcessingStatus(unittest.TestCase):
    """Test ProcessingStatus model."""

    def test_processing_status_creation(self):
        """Test processing status creation."""
        status = ProcessingStatus()
        self.assertFalse(status.detection_processed)
        self.assertFalse(status.tracking_processed)
        self.assertFalse(status.is_complete())

    def test_processing_status_complete(self):
        """Test processing status completion."""
        status = ProcessingStatus(
            detection_processed=True,
            tracking_processed=True,
            transformation_processed=True,
            team_assignment_processed=True,
            ball_assignment_processed=True,
            motion_analysis_processed=True,
        )
        self.assertTrue(status.is_complete())


class TestVideoMetadata(unittest.TestCase):
    """Test VideoMetadata model."""

    def test_video_metadata_creation(self):
        """Test video metadata creation."""
        metadata = VideoMetadata(
            file_path="/test/video.mp4",
            frame_rate=30.0,
            resolution=(1920, 1080),
            duration=60.0,
            codec="h264",
        )

        self.assertEqual(metadata.file_path, "/test/video.mp4")
        self.assertEqual(metadata.frame_rate, 30.0)
        self.assertEqual(metadata.resolution, (1920, 1080))
        self.assertEqual(metadata.duration, 60.0)
        self.assertEqual(metadata.codec, "h264")


class TestMatchContext(unittest.TestCase):
    """Test MatchContext model."""

    def test_match_context_creation(self):
        """Test match context creation."""
        context = MatchContext(
            match_id="match_001",
            home_team="Team A",
            away_team="Team B",
            final_score=(2, 1),
            venue="Stadium A",
        )

        self.assertEqual(context.match_id, "match_001")
        self.assertEqual(context.home_team, "Team A")
        self.assertEqual(context.away_team, "Team B")
        self.assertEqual(context.final_score, (2, 1))
        self.assertEqual(context.venue, "Stadium A")


if __name__ == "__main__":
    unittest.main()
