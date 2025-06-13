import sys
from typing import Dict, List, Any

import cv2
import numpy as np

sys.path.append("../")
from utils import measure_distance, get_foot_position


class SpeedAndDistanceEstimator:
    """
    A class for calculating and visualizing speed and distance metrics for tracked objects.

    This class processes tracking data to compute speed (km/h) and cumulative distance (meters)
    for players, excluding balls and referees. It uses transformed coordinates to ensure
    accurate real-world measurements.
    """

    def __init__(self, frame_window: int = 5, frame_rate: int = 24) -> None:
        """
        Initialize the SpeedAndDistanceEstimator.

        Args:
            frame_window (int): Number of frames to use for speed calculation window (default: 5)
            frame_rate (int): Video frame rate in frames per second (default: 24)
        """
        self.frame_window = frame_window
        self.frame_rate = frame_rate

    def add_speed_and_distance_to_tracks(
        self, tracks: Dict[str, List[Dict[str, Dict[str, Any]]]]
    ) -> None:
        """
        Calculate and add speed and distance metrics to tracking data.

        This method processes tracking data to compute speed and cumulative distance for each
        tracked object. It uses a sliding window approach to calculate instantaneous speed
        and maintains running totals for distance traveled.

        Args:
            tracks (Dict[str, List[Dict[str, Dict[str, Any]]]]): Nested dictionary structure containing:
                - object type (str) -> list of frames
                - each frame -> dict of track_id -> track_info
                - track_info must contain 'position_transformed' and will be updated with 'speed' and 'distance'
        """
        total_distance: Dict[str, Dict[str, float]] = {}

        for object_type, object_tracks in tracks.items():
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
                        if track_id not in tracks[object_type][frame_num_batch]:
                            continue
                        tracks[object_type][frame_num_batch][track_id][
                            "speed"
                        ] = speed_km_per_hour
                        tracks[object_type][frame_num_batch][track_id]["distance"] = (
                            total_distance[object_type][track_id]
                        )

    def draw_speed_and_distance(
        self,
        frames: List[np.ndarray],
        tracks: Dict[str, List[Dict[str, Dict[str, Any]]]],
    ) -> List[np.ndarray]:
        """
        Draw speed and distance information on video frames.

        This method overlays speed (km/h) and cumulative distance (meters) text on each frame
        for tracked objects that have speed data available.

        Args:
            frames (List[np.ndarray]): List of video frames as numpy arrays
            tracks (Dict[str, List[Dict[str, Dict[str, Any]]]]): Tracking data containing speed and distance info

        Returns:
            List[np.ndarray]: List of frames with speed and distance annotations drawn
        """
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object_type, object_tracks in tracks.items():
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

        return output_frames
