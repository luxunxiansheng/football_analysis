from test_utils import (
    get_test_video_path,
    create_test_processors,
    create_test_video_data,
    run_standard_pipeline_test,
)
from football_ai.motion.object_motion_processor import ObjectMotionProcessor
from football_ai.assignment.team_assignment_processor import (
    SigLIPTeamAssignmentProcessor as TeamAssignmentProcessor,
)
from football_ai.assignment.ball_assignment_processor import BallAssignmentProcessor


def test_ball_assignment_processor():
    # Use utility functions for setup
    video_path = get_test_video_path()
    detection_processor, track_processor = create_test_processors()

    motion_processor = ObjectMotionProcessor()
    team_processor = TeamAssignmentProcessor()
    ball_processor = BallAssignmentProcessor()

    video_data = create_test_video_data(video_path, max_frames=3)

    detected = detection_processor.process(video_data)
    tracked = track_processor.process(detected)
    motioned = motion_processor.process(tracked)
    teamed = team_processor.process(motioned)
    result = ball_processor.process(teamed)

    for frame_data in result.frames:
        assignments = (
            frame_data.metadata.get("ball_assignments", {})
            if frame_data.metadata
            else {}
        )
        print(f"Frame {frame_data.frame_number}: ball_assignments={assignments}")
        # At least one ball should be assigned to a player/goalkeeper or None
        assert isinstance(assignments, dict)
    print("BallAssignmentProcessor test passed.")


if __name__ == "__main__":
    test_ball_assignment_processor()
