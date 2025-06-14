import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

try:
    import supervision as sv
except ImportError:
    sv = None

# Local application imports
sys.path.append("../")
from camera_movement_estimator import CameraMovementEstimator
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from view_transformer import ViewTransformer
from utils.bbox_utils import (
    get_center_of_bbox,
    get_bbox_width,
    get_foot_position,
    measure_distance,
)


class TrackManager:
    """
    Manages object tracking, team assignment, and video annotation for football analysis.

    This class coordinates multiple components to track players, referees, and the ball
    across video frames, assigns teams based on jersey colors, and provides visualization.
    """

    def __init__(
        self,
        model_path: str,
        team_assigner: TeamAssigner,
        player_ball_assigner: PlayerBallAssigner,
        camera_movement_estimator: CameraMovementEstimator,
        view_transformer: ViewTransformer,
        frame_window: int = 5,
        frame_rate: int = 24,
    ):
        """
        Initialize the TrackManager with required components.

        Args:
            model_path: Path to YOLO model file
            team_assigner: Component for assigning players to teams
            player_ball_assigner: Component for ball-player assignment
            camera_movement_estimator: Component for camera movement estimation
            view_transformer: Component for coordinate transformation
            frame_window: Window size for speed/distance calculations
            frame_rate: Video frame rate for speed calculations
        """
        self.model = YOLO(model_path)
        self.frames: Optional[List[np.ndarray]] = None
        self.tracks = {"players": [], "referees": [], "ball": []}
        self.camera_movement_per_frame: Optional[List[List[float]]] = None
        self.team_ball_control: Optional[np.ndarray] = None

        self.team_assigner = team_assigner
        self.player_ball_assigner = player_ball_assigner
        self.camera_movement_estimator = camera_movement_estimator
        self.view_transformer = view_transformer

        self.frame_window = frame_window
        self.frame_rate = frame_rate

    def initialize(
        self,
        tracker: Any,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> None:
        """
        Initialize tracking with frames and optional cached data.

        Args:
            tracker: ByteTrack tracker instance
            frames: List of video frames
            read_from_stub: Whether to load cached tracking data
            stub_path: Path to cached tracking data file
        """
        self.frames = frames

        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                self.tracks = pickle.load(f)
                return

        detections = self._detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision Detection format
            try:
                detection_supervision = sv.Detections.from_ultralytics(detection)

                # Convert GoalKeeper to player object
                if detection_supervision.class_id is not None:
                    for object_ind, class_id in enumerate(
                        detection_supervision.class_id
                    ):
                        if cls_names[class_id] == "goalkeeper":
                            detection_supervision.class_id[object_ind] = cls_names_inv[
                                "player"
                            ]

                # Track Objects
                detection_with_tracks = tracker.update_with_detections(
                    detection_supervision
                )
            except (AttributeError, ImportError):
                # Fallback if supervision is not available
                detection_with_tracks = []
                detection_supervision = None

            self.tracks["players"].append({})
            self.tracks["referees"].append({})
            self.tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    self.tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv["referee"]:
                    self.tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            if detection_supervision is not None:
                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]

                    if cls_id == cls_names_inv["ball"]:
                        self.tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(self.tracks, f)

    def assign_team_colors(self) -> None:
        """
        Assign team colors to players based on their jersey colors.
        Uses K-means clustering to determine the dominant jersey colors and assigns
        players to teams based on these colors.
        """
        if not self.frames or len(self.frames) == 0:
            print("Warning: No frames available for team color assignment")
            return

        self.team_assigner.assign_team_color(self.frames[0], self.tracks["players"][0])
        for frame_num, player_track in enumerate(self.tracks["players"]):
            for player_id, track in player_track.items():
                team = self.team_assigner.get_player_team(
                    self.frames[frame_num], track["bbox"], player_id
                )
                self.tracks["players"][frame_num][player_id]["team"] = team
                self.tracks["players"][frame_num][player_id]["team_color"] = (
                    self.team_assigner.team_colors[team]
                )

    def add_positions(self) -> None:
        for object, object_tracks in self.tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    self.tracks[object][frame_num][track_id]["position"] = position

    def add_adjust_positions(self) -> None:
        if self.frames is None:
            raise ValueError("Frames must be initialized before adjusting positions")

        self.camera_movement_per_frame = (
            self.camera_movement_estimator.get_camera_movement(
                self.frames, read_from_stub=True
            )
        )

        for object_type, object_tracks in self.tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position: Tuple[float, float] = track_info["position"]
                    camera_movement: List[float] = self.camera_movement_per_frame[
                        frame_num
                    ]
                    position_adjusted: Tuple[float, float] = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1],
                    )
                    self.tracks[object_type][frame_num][track_id][
                        "position_adjusted"
                    ] = position_adjusted

    def assign_ball_to_players(self) -> None:
        team_ball_control = []
        for frame_num, player_track in enumerate(self.tracks["players"]):
            ball_bbox = self.tracks["ball"][frame_num][1]["bbox"]
            assigned_player = self.player_ball_assigner.assign_ball_to_player(
                player_track, ball_bbox
            )

            if assigned_player != -1:
                self.tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_ball_control.append(
                    self.tracks["players"][frame_num][assigned_player]["team"]
                )
            else:
                team_ball_control.append(team_ball_control[-1])
        self.team_ball_control = np.array(team_ball_control)

    def add_transformed_position(self) -> None:

        for object, object_tracks in self.tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position_adjusted"]
                    position = np.array(position)
                    position_transformed = self.view_transformer.transform_point(
                        position
                    )
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()

                    self.tracks[object][frame_num][track_id][
                        "position_transformed"
                    ] = position_transformed

    def interpolate_ball_positions(self) -> None:
        ball_positions = [x.get(1, {}).get("bbox", []) for x in self.tracks["ball"]]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()
        ]

        self.tracks["ball"] = ball_positions

    def add_speed_and_distance(self) -> None:

        total_distance: Dict[str, Dict[str, float]] = {}

        for object_type, object_tracks in self.tracks.items():
            if object_type == "ball" or object_type == "referees":
                continue

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id][
                        "position_transformed"
                    ]
                    end_position = object_tracks[last_frame][track_id][
                        "position_transformed"
                    ]

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    if object_type not in total_distance:
                        total_distance[object_type] = {}

                    if track_id not in total_distance[object_type]:
                        total_distance[object_type][track_id] = 0

                    total_distance[object_type][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in self.tracks[object_type][frame_num_batch]:
                            continue
                        self.tracks[object_type][frame_num_batch][track_id][
                            "speed"
                        ] = speed_km_per_hour
                        self.tracks[object_type][frame_num_batch][track_id][
                            "distance"
                        ] = total_distance[object_type][track_id]

    def draw_annotations(self) -> None:
        """Draw tracking annotations on frames."""
        if not self.frames:
            return

        output_frames = []
        for frame_num, frame in enumerate(self.frames):
            frame = frame.copy()

            player_dict = self.tracks["players"][frame_num]
            ball_dict = self.tracks["ball"][frame_num]
            referee_dict = self.tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self._draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self._draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self._draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self._draw_traingle(frame, ball["bbox"], (0, 255, 0))

            output_frames.append(frame)

        self.frames = output_frames

    def _detect_frames(
        self, frames: List[np.ndarray], batch_size: int = 20, conf: float = 0.1
    ) -> List[Any]:
        """Run YOLO detection on frames in batches."""
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i + batch_size], conf=conf)
            detections += detections_batch
        return detections

    def _draw_ellipse(
        self,
        frame: np.ndarray,
        bbox: List[float],
        color: Tuple[int, int, int],
        track_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw an ellipse around tracked objects (players/referees) at foot level.

        Creates an ellipse at the bottom of the bounding box to represent the player's
        ground position, with an optional track ID label.

        Args:
            frame (np.ndarray): Video frame to draw on
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2]
            color (Tuple[int, int, int]): BGR color for the ellipse (B, G, R)
            track_id (Optional[int]): Track ID to display in label

        Returns:
            np.ndarray: Frame with ellipse and label drawn

        Note:
            - Ellipse is positioned at the bottom center of bbox
            - Rectangle label adjusts width for track_id > 99
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def _draw_traingle(
        self, frame: np.ndarray, bbox: List[float], color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Draw a triangle marker above tracked objects (typically for ball or special indicators).

        Creates a triangle pointing downward at the top of the bounding box, commonly
        used to mark the ball or indicate which player has possession.

        Args:
            frame (np.ndarray): Video frame to draw on
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2]
            color (Tuple[int, int, int]): BGR color for the triangle (B, G, R)

        Returns:
            np.ndarray: Frame with triangle drawn

        Note:
            Triangle is positioned at top center of bbox with black outline
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ]
        )
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_camera_movement(self) -> None:
        """Draw camera movement information on frames."""
        if not self.frames or not self.camera_movement_per_frame:
            return

        output_frames = []
        for frame_num, frame in enumerate(self.frames):
            frame = frame.copy()

            # Create semi-transparent overlay for text background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Add camera movement text
            x_movement, y_movement = self.camera_movement_per_frame[frame_num]
            frame = cv2.putText(
                frame,
                f"Camera Movement X: {x_movement:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )
            frame = cv2.putText(
                frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )

            output_frames.append(frame)

        self.frames = output_frames

    def draw_team_ball_control(self) -> None:
        """
        Draw team ball control statistics overlay on the video frame.

        Creates a semi-transparent panel showing the percentage of time each team
        has controlled the ball up to the current frame.
        """
        if not self.frames or self.team_ball_control is None:
            return

        output_frames = []
        for frame_num, frame in enumerate(self.frames):
            frame = frame.copy()
            # Draw a semi-transparent rectangle
            overlay = frame.copy()
            cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            team_ball_control_till_frame = self.team_ball_control[: frame_num + 1]
            # Get the number of time each team had ball control
            team_1_num_frames = team_ball_control_till_frame[
                team_ball_control_till_frame == 1
            ].shape[0]
            team_2_num_frames = team_ball_control_till_frame[
                team_ball_control_till_frame == 2
            ].shape[0]

            total_frames = team_1_num_frames + team_2_num_frames
            if total_frames > 0:
                team_1 = team_1_num_frames / total_frames
                team_2 = team_2_num_frames / total_frames
            else:
                team_1 = team_2 = 0

            cv2.putText(
                frame,
                f"Team 1 Ball Control: {team_1*100:.2f}%",
                (1400, 900),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                frame,
                f"Team 2 Ball Control: {team_2*100:.2f}%",
                (1400, 950),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )
            output_frames.append(frame)

        self.frames = output_frames

    def draw_speed_and_distance(self) -> None:
        """
        Draw speed and distance information on video frames.

        This method overlays speed (km/h) and cumulative distance (meters) text on each frame
        for tracked objects that have speed data available.
        """
        if not self.frames:
            return

        output_frames = []
        for frame_num, frame in enumerate(self.frames):
            for object_type, object_tracks in self.tracks.items():
                if object_type == "ball" or object_type == "referees":
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get("speed", None)
                        distance = track_info.get("distance", None)
                        if speed is None or distance is None:
                            continue

                        bbox = track_info["bbox"]
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(
                            frame,
                            f"{speed:.2f} km/h",
                            position,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"{distance:.2f} m",
                            (position[0], position[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2,
                        )
            output_frames.append(frame)

        self.frames = output_frames
