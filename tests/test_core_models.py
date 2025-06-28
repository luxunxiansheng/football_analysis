import unittest
from football_ai.core_models.player import Player
from football_ai.core_models.goalkeeper import Goalkeeper
from football_ai.core_models.referee import Referee
from football_ai.core_models.ball import Ball
from football_ai.core_models.frame import CameraMotion, ProcessingStatus
from football_ai.core_models.field import Field


class TestPlayer(unittest.TestCase):
    def test_player_fields(self):
        player = Player(
            track_id=1,
            player_id="p10",
            jersey_number=10,
            team_id=2,
            pixel_position=(100.0, 200.0),
            field_position=(50.0, 30.0),
            speed=7.5,
            direction=90.0,
            acceleration=1.2,
            detection_confidence=0.95,
            track_confidence=0.9,
            track_age=15,
            last_seen_frame=100,
            is_active=True,
            total_distance=1200.0,
            max_speed=9.8,
            avg_speed=7.0,
            sprint_count=3,
            position_history=[(100.0, 200.0), (101.0, 201.0)],
            field_position_history=[(50.0, 30.0), (51.0, 31.0)],
        )
        self.assertEqual(player.track_id, 1)
        self.assertEqual(player.jersey_number, 10)
        self.assertEqual(player.team_id, 2)
        self.assertEqual(player.pixel_position, (100.0, 200.0))
        self.assertEqual(player.speed, 7.5)
        self.assertEqual(player.detection_confidence, 0.95)
        self.assertEqual(player.position_history, [(100.0, 200.0), (101.0, 201.0)])


class TestGoalkeeper(unittest.TestCase):
    def test_goalkeeper_fields(self):
        keeper = Goalkeeper(
            track_id=2,
            player_id="gk1",
            jersey_number=1,
            team_id=1,
            pixel_position=(10.0, 20.0),
            field_position=(5.0, 3.0),
            speed=5.0,
            direction=180.0,
            acceleration=0.8,
            detection_confidence=0.99,
            track_confidence=0.85,
            track_age=30,
            last_seen_frame=200,
            is_active=True,
            total_distance=800.0,
            max_speed=6.5,
            avg_speed=5.2,
            sprint_count=1,
            position_history=[(10.0, 20.0)],
            field_position_history=[(5.0, 3.0)],
        )
        self.assertEqual(keeper.jersey_number, 1)
        self.assertEqual(keeper.team_id, 1)
        self.assertEqual(keeper.speed, 5.0)
        self.assertEqual(keeper.detection_confidence, 0.99)


class TestReferee(unittest.TestCase):
    def test_referee_fields(self):
        ref = Referee(
            track_id=3,
            referee_id="r1",
            referee_type="main",
            pixel_position=(300.0, 400.0),
            field_position=(70.0, 40.0),
            speed=6.0,
            direction=45.0,
            acceleration=1.0,
            detection_confidence=0.92,
            track_confidence=0.8,
            track_age=10,
            last_seen_frame=150,
            is_active=True,
            total_distance=1500.0,
            max_speed=7.2,
            avg_speed=6.1,
            position_history=[(300.0, 400.0)],
            field_position_history=[(70.0, 40.0)],
            zone_coverage="center",
        )
        self.assertEqual(ref.referee_type, "main")
        self.assertEqual(ref.zone_coverage, "center")
        self.assertEqual(ref.speed, 6.0)
        self.assertEqual(ref.detection_confidence, 0.92)


class TestBall(unittest.TestCase):
    def test_ball_fields(self):
        ball = Ball(
            track_id=4,
            ball_id="b1",
            pixel_position=(500.0, 600.0),
            field_position=(80.0, 50.0),
            speed=30.0,
            direction=270.0,
            acceleration=2.5,
            velocity_vector=(10.0, 20.0),
            ball_size=22.0,
            detection_confidence=0.98,
            track_confidence=0.97,
            track_age=5,
            last_seen_frame=300,
            is_active=True,
            is_visible=True,
            position_history=[(500.0, 600.0)],
            field_position_history=[(80.0, 50.0)],
            speed_history=[30.0, 29.5],
            controlling_player_id=1,
            last_touch_player_id=2,
            last_touch_frame=299,
        )
        self.assertEqual(ball.ball_id, "b1")
        self.assertEqual(ball.speed, 30.0)
        self.assertEqual(ball.velocity_vector, (10.0, 20.0))
        self.assertEqual(ball.is_visible, True)
        self.assertEqual(ball.controlling_player_id, 1)


class TestFrameAndField(unittest.TestCase):
    def test_camera_motion(self):
        cam = CameraMotion(x_offset=1.0, y_offset=2.0, rotation=0.5, zoom=1.1)
        self.assertTrue(cam)
        cam2 = CameraMotion()
        self.assertFalse(cam2)

    def test_processing_status(self):
        status = ProcessingStatus(
            detection_processed=True,
            tracking_processed=True,
            transformation_processed=True,
            team_assignment_processed=True,
            ball_assignment_processed=True,
            motion_analysis_processed=True,
            rendering_processed=True,
        )
        self.assertTrue(status.is_complete())
        status2 = ProcessingStatus()
        self.assertFalse(status2.is_complete())

    def test_field_fields(self):
        field = Field(
            field_id="f1",
            field_name="Main Field",
            field_type="football",
            length=105.0,
            width=68.0,
            goal_width=7.32,
            goal_height=2.44,
            goal_depth=2.0,
            penalty_area_length=16.5,
            penalty_area_width=40.32,
            goal_area_length=5.5,
            goal_area_width=18.32,
            center_circle_radius=9.15,
            field_corners=[(0.0, 0.0), (0.0, 68.0), (105.0, 0.0), (105.0, 68.0)],
            field_boundaries=[(0.0, 0.0), (0.0, 68.0), (105.0, 0.0), (105.0, 68.0)],
            left_goal_posts=[(0.0, 30.0), (0.0, 38.0), (1.0, 30.0), (1.0, 38.0)],
            right_goal_posts=[
                (105.0, 30.0),
                (105.0, 38.0),
                (104.0, 30.0),
                (104.0, 38.0),
            ],
            center_line=[(52.5, 0.0), (52.5, 68.0)],
            left_penalty_area=[(0.0, 20.0), (16.5, 20.0), (16.5, 48.0), (0.0, 48.0)],
            right_penalty_area=[
                (105.0, 20.0),
                (88.5, 20.0),
                (88.5, 48.0),
                (105.0, 48.0),
            ],
            left_goal_area=[(0.0, 30.0), (5.5, 30.0), (5.5, 38.0), (0.0, 38.0)],
            right_goal_area=[(105.0, 30.0), (99.5, 30.0), (99.5, 38.0), (105.0, 38.0)],
        )
        self.assertEqual(field.field_id, "f1")
        self.assertEqual(field.length, 105.0)
        self.assertEqual(field.goal_width, 7.32)
        self.assertEqual(field.center_circle_radius, 9.15)
        self.assertEqual(
            field.field_corners, [(0.0, 0.0), (0.0, 68.0), (105.0, 0.0), (105.0, 68.0)]
        )


if __name__ == "__main__":
    unittest.main()
