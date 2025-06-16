import numpy as np
from ..domain.data_models import VideoData, FrameData
from ..domain.interfaces import Processor
import cv2


class CameraMotionProcessor(Processor):
    def __init__(self):
        self.prev_gray = None
        self.prev_pts = None
        self.movements = []

    def process(self, data: VideoData) -> VideoData:
        for frame_data in data.frames:
            frame = frame_data.raw_frame
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is not None:
                # Detect good features to track in previous frame
                prev_pts = cv2.goodFeaturesToTrack(
                    self.prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7
                )
                if prev_pts is not None:
                    # Calculate optical flow (track feature points)
                    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, prev_pts, None
                    )
                    # Compute movement as mean displacement of tracked points
                    if next_pts is not None and status is not None:
                        movement = np.mean(
                            next_pts[status.flatten() == 1]
                            - prev_pts[status.flatten() == 1],
                            axis=0,
                        )
                        if frame_data.metadata is None:
                            frame_data.metadata = {}
                        frame_data.metadata["camera_movement"] = movement.tolist()
                        self.movements.append(movement)
                    else:
                        if frame_data.metadata is None:
                            frame_data.metadata = {}
                        frame_data.metadata["camera_movement"] = [0.0, 0.0]
                else:
                    if frame_data.metadata is None:
                        frame_data.metadata = {}
                    frame_data.metadata["camera_movement"] = [0.0, 0.0]
            else:
                if frame_data.metadata is None:
                    frame_data.metadata = {}
                frame_data.metadata["camera_movement"] = [0.0, 0.0]
            self.prev_gray = gray
        return data
