import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.append("../")
from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator:
    """
    Estimates camera movement between frames using optical flow tracking.

    This class uses Lucas-Kanade optical flow to track feature points and
    determine camera movement between consecutive frames. It can adjust
    object positions based on camera movement and visualize the movement.
    """

    def __init__(self, frame: np.ndarray) -> None:
        """
        Initialize the camera movement estimator with the first frame.

        Args:
            frame: First frame of the video sequence as numpy array
        """
        self.minimum_distance: int = 5

        # Lucas-Kanade optical flow parameters
        self.lk_params: Dict[str, Any] = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Create mask for feature detection (focus on edges of frame)
        first_frame_grayscale: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features: np.ndarray = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1  # Left edge
        mask_features[:, 900:1050] = 1  # Right edge

        # Good features to track parameters
        self.features: Dict[str, Any] = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def add_adjust_positions_to_tracks(
        self,
        tracks: Dict[str, List[Dict[int, Dict[str, Any]]]],
        camera_movement_per_frame: List[List[float]],
    ) -> None:
        """
        Adjust object positions in tracks based on camera movement.

        Args:
            tracks: Dictionary containing tracking data for different objects
            camera_movement_per_frame: List of [x, y] camera movements per frame
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position: Tuple[float, float] = track_info["position"]
                    camera_movement: List[float] = camera_movement_per_frame[frame_num]
                    position_adjusted: Tuple[float, float] = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1],
                    )
                    tracks[object_type][frame_num][track_id][
                        "position_adjusted"
                    ] = position_adjusted

    def get_camera_movement(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Calculate camera movement for each frame using optical flow.

        Args:
            frames: List of video frames as numpy arrays
            read_from_stub: Whether to read cached results from file
            stub_path: Path to cache file for saving/loading results

        Returns:
            List of [x, y] camera movements for each frame
        """
        # Read from cache if available
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        # Initialize camera movement array
        camera_movement: List[List[float]] = [[0, 0]] * len(frames)

        # Convert first frame to grayscale and detect features
        old_gray: np.ndarray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features: np.ndarray = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Process each subsequent frame
        for frame_num in range(1, len(frames)):
            frame_gray: np.ndarray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            max_distance: float = 0
            camera_movement_x: float = 0
            camera_movement_y: float = 0

            # Find the feature point with maximum movement
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point: np.ndarray = new.ravel()
                old_features_point: np.ndarray = old.ravel()

                distance: float = measure_distance(
                    new_features_point, old_features_point
                )
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_features_point, new_features_point
                    )

            # Update camera movement if significant movement detected
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        # Cache results if path provided
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(
        self, frames: List[np.ndarray], camera_movement_per_frame: List[List[float]]
    ) -> List[np.ndarray]:
        """
        Draw camera movement information on frames.

        Args:
            frames: List of video frames as numpy arrays
            camera_movement_per_frame: List of [x, y] camera movements per frame

        Returns:
            List of frames with camera movement information overlaid
        """
        output_frames: List[np.ndarray] = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            # Create semi-transparent overlay for text background
            overlay: np.ndarray = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha: float = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Add camera movement text
            x_movement, y_movement = camera_movement_per_frame[frame_num]
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

        return output_frames
