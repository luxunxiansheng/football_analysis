from ..domain.data_models import VideoData
from ..domain.interfaces import Processor
from tqdm import tqdm
import warnings

# Suppress sklearn FutureWarning about 'force_all_finite'
warnings.filterwarnings(
    "ignore", message=".*force_all_finite.*", category=FutureWarning
)


class BallControlProcessor(Processor):
    """
    Calculates each team's ball control percentage based on the number of frames
    where the ball is in possession of each team.
    """

    def __init__(self):
        self._ball_control_counts = {}  # team_id -> frame_count
        self._total_frames_with_ball = 0

    def process(self, data: VideoData) -> VideoData:
        """
        Process video data to calculate ball control statistics.

        Args:
            data: VideoData containing frames with detections and ball assignments

        Returns:
            VideoData with ball control metadata added
        """
        # Reset counters
        self._ball_control_counts = {}
        self._total_frames_with_ball = 0

        # Use progress bar only if processing many frames (>50)
        if len(data.frames) > 50:
            progress_bar = tqdm(
                data.frames, desc="Calculating ball control", unit="frames"
            )
            frame_iterator = progress_bar
        else:
            frame_iterator = data.frames
            progress_bar = None

        for frame_data in frame_iterator:
            # Process frame and update running totals
            self._process_frame(frame_data)

            # Store basic ball control data in frame's ball_control object
            # For now, use simplified approach - you can enhance this later
            controlling_player = None
            for player_id, count in self._ball_control_counts.items():
                if count > 0:  # Simple logic - can be enhanced
                    controlling_player = player_id
                    break

            frame_data.ball_control.controlling_player = controlling_player

        if progress_bar:
            progress_bar.close()

        # Also add final stats to video custom data
        if data.custom is None:
            data.custom = {}
        data.custom["ball_control"] = {
            "percentages": self._calculate_percentages(),
            "frame_counts": self._ball_control_counts.copy(),
            "total_frames_analyzed": self._total_frames_with_ball,
            "total_video_frames": len(data.frames),
        }

        # Debug: Confirm video metadata is stored
        print(f"DEBUG: Video custom data stored - {data.custom['ball_control']}")

        return data

    def _process_frame(self, frame_data):
        """Process a single frame to update ball control counts."""
        if not frame_data.detections:
            return

        detections = frame_data.detections

        # Debug: Count detection types
        balls_found = 0
        balls_with_players = 0

        # Find balls with assigned players
        for detection in detections:
            if self._is_ball(detection):
                balls_found += 1
                if self._has_assigned_player(detection):
                    balls_with_players += 1
                    assigned_idx = detection.metadata["assigned_player"]
                    if assigned_idx < len(detections):
                        player = detections[assigned_idx]
                        team_id = self._get_team_id(player)

                        if team_id is not None:
                            if team_id not in self._ball_control_counts:
                                self._ball_control_counts[team_id] = 0
                            self._ball_control_counts[team_id] += 1
                            self._total_frames_with_ball += 1
                            break  # Only count one ball per frame

        # Debug output every 100 frames
        if self._total_frames_with_ball % 100 == 0 and self._total_frames_with_ball > 0:
            print(
                f"DEBUG: Frame processed - balls found: {balls_found}, balls with players: {balls_with_players}"
            )
            print(f"DEBUG: Current counts: {self._ball_control_counts}")
            print(f"DEBUG: Total frames: {self._total_frames_with_ball}")

    def _is_ball(self, detection):
        """Check if detection is a ball."""
        return (
            getattr(detection, "object_type", None) == "ball"
            or getattr(detection, "class_name", "").lower() == "ball"
        )

    def _has_assigned_player(self, ball_detection):
        """Check if ball has an assigned player."""
        return (
            hasattr(ball_detection, "metadata")
            and ball_detection.metadata
            and "assigned_player" in ball_detection.metadata
            and ball_detection.metadata["assigned_player"] is not None
        )

    def _get_team_id(self, player_detection):
        """Get team ID from player detection."""
        if hasattr(player_detection, "metadata") and player_detection.metadata:
            for field in ["team_id", "team", "cluster_id"]:
                if field in player_detection.metadata:
                    return player_detection.metadata[field]
        return None

    def _calculate_percentages(self):
        """Calculate ball control percentages for each team."""
        if self._total_frames_with_ball == 0:
            return {}

        percentages = {}
        for team_id, frame_count in self._ball_control_counts.items():
            percentages[team_id] = (frame_count / self._total_frames_with_ball) * 100.0

        return percentages

    def print_debug_info(self):
        """Print debug information about ball control processing."""
        print(f"DEBUG: Total frames with ball: {self._total_frames_with_ball}")
        print(f"DEBUG: Ball control counts: {self._ball_control_counts}")
        print(f"DEBUG: Percentages: {self._calculate_percentages()}")

    def get_video_level_stats(self, data: VideoData):
        """Get video-level ball control statistics."""
        if not data.custom or "ball_control" not in data.custom:
            return None
        return data.custom["ball_control"]

    def print_video_summary(self, data: VideoData):
        """Print a summary of video-level ball control statistics."""
        stats = self.get_video_level_stats(data)
        if not stats:
            print("No video-level ball control data available.")
            return

        print("\n" + "=" * 50)
        print("VIDEO-LEVEL BALL CONTROL SUMMARY")
        print("=" * 50)
        print(f"Total video frames: {stats.get('total_video_frames', 0)}")
        print(f"Frames with ball possession: {stats.get('total_frames_analyzed', 0)}")

        percentages = stats.get("percentages", {})
        if percentages:
            print("\nFINAL POSSESSION PERCENTAGES:")
            for team_id in sorted(percentages.keys()):
                percentage = percentages[team_id]
                frames = stats.get("frame_counts", {}).get(team_id, 0)
                print(f"  Team {team_id}: {percentage:.1f}% ({frames} frames)")
        else:
            print("\nNo possession detected in video.")
        print("=" * 50)
