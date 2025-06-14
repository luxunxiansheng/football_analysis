import os
import pickle
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

sys.path.append("../")
from utils.bbox_utils import measure_distance, measure_xy_distance


class CameraMovementEstimator:
    """
    Estimates camera movement between frames using optical flow tracking.

    This class uses Lucas-Kanade optical flow to track feature points and
    determine camera movement between consecutive frames. It can adjust
    object positions based on camera movement and visualize the movement.
    """

    def __init__(self, stub_path: Optional[str] = None) -> None:
        """
        Initialize the camera movement estimator.

        Args:
            stub_path: Optional path to cache camera movement data
        """
        self.minimum_distance = 5
        self.stub_path = stub_path

        # Lucas-Kanade optical flow parameters
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }

    def get_camera_movement(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
    ) -> List[List[float]]:
        """
        Estimate camera movement between frames using optical flow.

        Args:
            frames: List of video frames
            read_from_stub: Whether to read from cached file if available

        Returns:
            List of [x, y] camera movements per frame
        """
        # Create mask for feature detection (focus on edges of frame)
        first_frame_grayscale = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1  # Left edge
        mask_features[:, 900:1050] = 1  # Right edge

        # Good features to track parameters
        features_params = {
            "maxCorners": 100,
            "qualityLevel": 0.3,
            "minDistance": 3,
            "blockSize": 7,
            "mask": mask_features,
        }

        # Read from cache if available
        if read_from_stub and self.stub_path and os.path.exists(self.stub_path):
            with open(self.stub_path, "rb") as f:
                return pickle.load(f)

        # Initialize camera movement array
        camera_movement: List[List[float]] = [[0.0, 0.0] for _ in range(len(frames))]

        # Convert first frame to grayscale and detect features
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **features_params)

        # Process each subsequent frame
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            if old_features is not None:
                new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, old_features, np.array([]), **self.lk_params
                )

                max_distance = 0.0
                camera_movement_x = 0.0
                camera_movement_y = 0.0

                # Find the feature point with maximum movement
                if new_features is not None:
                    for new, old in zip(new_features, old_features):
                        new_point = new.ravel()
                        old_point = old.ravel()

                        distance = measure_distance(new_point, old_point)
                        if distance > max_distance:
                            max_distance = distance
                            camera_movement_x, camera_movement_y = measure_xy_distance(
                                old_point, new_point
                            )

                # Update camera movement if significant movement detected
                if max_distance > self.minimum_distance:
                    camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                    old_features = cv2.goodFeaturesToTrack(
                        frame_gray, **features_params
                    )

            old_gray = frame_gray.copy()

        # Cache results if path provided
        if self.stub_path is not None:
            with open(self.stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement
