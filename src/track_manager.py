import os
import pickle
import sys
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from ultralytics import YOLO
import supervision as sv


# Local application imports
sys.path.append("../")
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from utils.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position


class TrackManager:
    def __init__(
        self,
        model_path: str,
        team_assigner: TeamAssigner,
        player_ball_assigner: PlayerBallAssigner,
    ):
        self.model: YOLO = YOLO(model_path)
        self.frames = None
        self.tracks = {"players": [], "referees": [], "ball": []}
        self.team_assigner = team_assigner
        self.player_ball_assigner = player_ball_assigner

    def initialize(
        self,
        tracker: sv.ByteTrack,  # Changed from Tracker to Any
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ):
        self.frames = frames

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                self.tracks = pickle.load(f)

        detections = self._detect_frames(self.frames)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = tracker.update_with_detections(
                detection_supervision
            )

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

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    self.tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(self.tracks, f)

    def assign_team_colors(
        self,
    ) -> None:
        """
        Assign team colors to players based on their jersey colors.
        Uses K-means clustering to determine the dominant jersey colors and assigns
        players to teams based on these colors.

        Args:
            frame (np.ndarray): The current video frame
            player_detections (List[dict]): List of player detection dictionaries
        """
        if self.frames is not None and len(self.frames) > 0:
            self.team_assigner.assign_team_color(
                self.frames[0], self.tracks["players"][0]
            )
        else:
            print("Warning: No frames available for team color assignment")

    def assign_ball_to_players(
        self,
     
    ) -> None:
        self.player_ball_assigner.assign_ball_to_players(
            self.tracks["players"][0], self.tracks["ball"][0][1]["bbox"]
        )
    def draw_team_ball_control(
        self,
        frame: np.ndarray,
        frame_num: int,
        team_ball_control: List[int],
    ) -> np.ndarray:
        """
        Draw the team ball control information on the frame.

        Args:
            frame (np.ndarray): The current video frame
            frame_num (int): The current frame number
            team_ball_control (List[int]): List indicating which team has ball control
                                           (1 for Team 1, 2 for Team 2)

        Returns:
            np.ndarray: The frame with team ball control information drawn
        """
        for track_id, player in self.tracks["players"][frame_num].items():
            if player.get("has_ball", False):
                team = team_ball_control[frame_num]
                color = (0, 255, 0) if team == 1 else (255, 0, 0)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

        return frame

    def draw_annotations(
        self,
    ) -> List[np.ndarray]:

        output_video_frames = []
        for frame_num, frame in enumerate(self.frames):
            frame = frame.copy()

            player_dict = self.tracks["players"][frame_num]
            ball_dict = self.tracks["ball"][frame_num]
            referee_dict = self.tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

    def _detect_frames(
        self, frames: List[np.ndarray], batch_size: int = 20, conf: float = 0.1
    ) -> List[Any]:
        """
        Perform object detection on a list of video frames using batch processing.

        Processes frames in batches to optimize GPU memory usage and inference speed.

        Args:
            frames (List[np.ndarray]): List of video frames (numpy arrays)

        Returns:
            List[Any]: List of YOLO detection results, one per frame

        Note:
            Uses batch_size=20 and confidence threshold=0.1 for detection
        """

        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i + batch_size], conf)
            detections += detections_batch
        return detections

    def draw_ellipse(
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

    def draw_traingle(
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

    def draw_team_ball_control(
        self, frame: np.ndarray, frame_num: int, team_ball_control: np.ndarray
    ) -> np.ndarray:
        """
        Draw team ball control statistics overlay on the video frame.

        Creates a semi-transparent panel showing the percentage of time each team
        has controlled the ball up to the current frame.

        Args:
            frame (np.ndarray): Video frame to draw on
            frame_num (int): Current frame number
            team_ball_control (np.ndarray): Array indicating which team (1 or 2)
                                             controls ball in each frame

        Returns:
            np.ndarray: Frame with ball control statistics overlay

        Note:
            - Overlay positioned at bottom-right of frame
            - Shows cumulative percentages from start to current frame
            - Uses white semi-transparent background
        """
        # Draw a semi-transparent rectaggle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[: frame_num + 1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[
            team_ball_control_till_frame == 1
        ].shape[0]
        team_2_num_frames = team_ball_control_till_frame[
            team_ball_control_till_frame == 2
        ].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

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

        return frame
