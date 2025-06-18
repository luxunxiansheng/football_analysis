from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.detection.core import Detections
import numpy as np
import logging
from tqdm import tqdm

from ..domain.data_models import VideoData
from ..domain.interfaces import Processor
from .track_optimizer import TrackIDOptimizer


class TrackProcessor(Processor):
    def __init__(
        self,
        track_activation_threshold: float = 0.15,
        lost_track_buffer: int = 120,
        minimum_matching_threshold: float = 0.95,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
        min_track_length: int = 5,
        max_merge_distance: float = 100.0,
        max_merge_frames: int = 25,
    ):
        """
        Initialize the TrackProcessor with explicit parameters.

        Args:
            track_activation_threshold: Minimum confidence to start a new track (default: 0.15)
            lost_track_buffer: Frames to keep lost tracks in memory (default: 120)
            minimum_matching_threshold: Minimum IoU for track matching (default: 0.95)
            frame_rate: Video frame rate for temporal calculations (default: 30)
            minimum_consecutive_frames: Minimum frames to confirm a track (default: 1)
            min_track_length: Minimum track length for optimization (default: 5)
            max_merge_distance: Maximum distance for merging tracks in pixels (default: 100.0)
            max_merge_frames: Maximum frame gap for merging tracks (default: 25)
        """
        # Store optimization parameters
        self.min_track_length = min_track_length
        self.max_merge_distance = max_merge_distance
        self.max_merge_frames = max_merge_frames

        # ByteTracker parameters optimized for football analysis
        self.tracker = ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

        self.logger = logging.getLogger(self.__class__.__name__)

    def process(self, data: VideoData) -> VideoData:
        # Use progress bar for tracking
        progress_bar = tqdm(data.frames, desc="Object tracking", unit="frames")

        for frame_data in progress_bar:
            detections = frame_data.detections or []
            if not detections:
                continue

            # More aggressive filtering to prevent spurious tracks
            # Different thresholds for different object types
            filtered_detections = []
            for detection in detections:
                min_conf = 0.3  # Default minimum confidence

                # Adjust confidence thresholds by object type
                if detection.object_type == "player":
                    min_conf = 0.4  # Higher for players (most important)
                elif detection.object_type == "ball":
                    min_conf = 0.2  # Lower for ball (harder to detect)
                elif detection.object_type == "referee":
                    min_conf = 0.35  # Medium for referees
                elif detection.object_type == "goalkeeper":
                    min_conf = 0.35  # Medium for goalkeepers

                if detection.confidence >= min_conf:
                    filtered_detections.append(detection)

            if not filtered_detections:
                # If no detections pass the filter, assign -1 to all
                for detection in detections:
                    if detection.metadata is None:
                        detection.metadata = {}
                    detection.metadata["track_id"] = -1
                continue

            boxes = np.array([det.bbox.as_list() for det in filtered_detections])
            confidences = np.array([det.confidence for det in filtered_detections])

            # Map object types to class IDs for better tracking
            def get_class_id(detection):
                if detection.object_type == "player":
                    return 0
                elif detection.object_type == "ball":
                    return 1
                elif detection.object_type == "referee":
                    return 2
                elif detection.object_type == "goalkeeper":
                    return 3
                else:
                    return 0  # Default to player

            class_ids = np.array([get_class_id(det) for det in filtered_detections])

            sv_detections = Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids,
            )

            tracked = self.tracker.update_with_detections(sv_detections)

            # Create a mapping of tracker IDs
            track_ids = getattr(
                tracked, "tracker_id", [None] * len(filtered_detections)
            )

            # Assign track IDs back to filtered detections
            for detection, tid in zip(filtered_detections, track_ids):
                if detection.metadata is None:
                    detection.metadata = {}
                detection.metadata["track_id"] = int(tid) if tid is not None else -1

            # For detections that were filtered out, assign -1 (untracked)
            for detection in detections:
                if detection not in filtered_detections:
                    if detection.metadata is None:
                        detection.metadata = {}
                    detection.metadata["track_id"] = -1

        progress_bar.close()

        # Post-process track IDs to optimize continuity and reduce fragmentation
        self.logger.info("Optimizing track IDs...")
        optimizer = TrackIDOptimizer(
            min_track_length=self.min_track_length,
            max_merge_distance=self.max_merge_distance,
            max_merge_frames=self.max_merge_frames,
        )
        data = optimizer.optimize_tracks(data)

        return data
