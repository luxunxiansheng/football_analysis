import numpy as np
from tqdm import tqdm
from football_ai.core_models import Video, Frame
from football_ai.core_models.interfaces import Processor
import cv2


class CameraMotionProcessor(Processor):
    def __init__(self):
        self.prev_gray = None
        self.prev_pts = None
        self.movements = []

    def process(self, data: Video) -> Video:
        # Use progress bar only if processing many frames (>100)
        if len(data.frames) > 100:
            progress_bar = tqdm(data.frames, desc="Camera motion", unit="frames")
            frame_iterator = progress_bar
        else:
            frame_iterator = data.frames
            progress_bar = None

        for frame_data in frame_iterator:
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
                    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore
                        self.prev_gray, gray, prev_pts, None
                    )
                    # Compute movement as mean displacement of tracked points
                    if next_pts is not None and status is not None:
                        try:
                            valid_mask = status.flatten() == 1
                            if np.any(valid_mask) and np.sum(valid_mask) > 0:
                                valid_prev = prev_pts[valid_mask]
                                valid_next = next_pts[valid_mask]
                                if len(valid_prev) > 0 and len(valid_next) > 0:
                                    movement = np.mean(valid_next - valid_prev, axis=0)
                                    # Ensure movement is valid and has correct shape
                                    movement = np.atleast_1d(
                                        movement
                                    )  # Ensure it's at least 1D
                                    if movement.size >= 2:
                                        frame_data.camera_motion.x_offset = movement[
                                            0
                                        ].item()
                                        frame_data.camera_motion.y_offset = movement[
                                            1
                                        ].item()
                                        self.movements.append(movement)
                                    elif movement.size == 1:
                                        # Single dimension case
                                        frame_data.camera_motion.x_offset = movement[
                                            0
                                        ].item()
                                        frame_data.camera_motion.y_offset = 0.0
                                    else:
                                        frame_data.camera_motion.x_offset = 0.0
                                        frame_data.camera_motion.y_offset = 0.0
                                else:
                                    frame_data.camera_motion.x_offset = 0.0
                                    frame_data.camera_motion.y_offset = 0.0
                            else:
                                frame_data.camera_motion.x_offset = 0.0
                                frame_data.camera_motion.y_offset = 0.0
                        except (ValueError, IndexError) as e:
                            # Handle array size/conversion errors
                            frame_data.camera_motion.x_offset = 0.0
                            frame_data.camera_motion.y_offset = 0.0
                    else:
                        frame_data.camera_motion.x_offset = 0.0
                        frame_data.camera_motion.y_offset = 0.0
                else:
                    frame_data.camera_motion.x_offset = 0.0
                    frame_data.camera_motion.y_offset = 0.0
            else:
                frame_data.camera_motion.x_offset = 0.0
                frame_data.camera_motion.y_offset = 0.0
            self.prev_gray = gray

        if progress_bar:
            progress_bar.close()
        return data
