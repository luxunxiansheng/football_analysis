from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.detection.core import Detections
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Dict, Optional
import cv2

from ..domain.data_models import VideoData, Detection
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
        # Enhanced filtering parameters
        enable_advanced_filtering: bool = True,
        spatial_validation: bool = True,
        temporal_validation: bool = True,
        size_validation: bool = True,
        adaptive_thresholds: bool = True,
        max_speed_threshold: float = 15.0,  # m/s
        min_size_threshold: float = 0.0001,  # relative to frame area
        max_size_threshold: float = 0.1,  # relative to frame area
        stability_window: int = 5,  # frames for stability check
    ):
        """
        Initialize the Enhanced TrackProcessor with advanced filtering capabilities.

        Args:
            track_activation_threshold: Minimum confidence to start a new track (default: 0.15)
            lost_track_buffer: Frames to keep lost tracks in memory (default: 120)
            minimum_matching_threshold: Minimum IoU for track matching (default: 0.95)
            frame_rate: Video frame rate for temporal calculations (default: 30)
            minimum_consecutive_frames: Minimum frames to confirm a track (default: 1)
            min_track_length: Minimum track length for optimization (default: 5)
            max_merge_distance: Maximum distance for merging tracks in pixels (default: 100.0)
            max_merge_frames: Maximum frame gap for merging tracks (default: 25)
            enable_advanced_filtering: Enable advanced filtering logic (default: True)
            spatial_validation: Enable spatial validation (default: True)
            temporal_validation: Enable temporal validation (default: True)
            size_validation: Enable size validation (default: True)
            adaptive_thresholds: Enable adaptive thresholds (default: True)
            max_speed_threshold: Maximum realistic speed in m/s (default: 15.0)
            min_size_threshold: Minimum detection size relative to frame (default: 0.0001)
            max_size_threshold: Maximum detection size relative to frame (default: 0.1)
            stability_window: Number of frames for stability checks (default: 5)
        """
        # Store optimization parameters
        self.min_track_length = min_track_length
        self.max_merge_distance = max_merge_distance
        self.max_merge_frames = max_merge_frames

        # Enhanced filtering parameters
        self.enable_advanced_filtering = enable_advanced_filtering
        self.spatial_validation = spatial_validation
        self.temporal_validation = temporal_validation
        self.size_validation = size_validation
        self.adaptive_thresholds = adaptive_thresholds
        self.max_speed_threshold = max_speed_threshold
        self.min_size_threshold = min_size_threshold
        self.max_size_threshold = max_size_threshold
        self.stability_window = stability_window

        # Track history for advanced validation
        self.track_history: Dict[int, List[Dict]] = {}
        self.frame_count = 0
        self.frame_dimensions = None

        # ByteTracker parameters optimized for football analysis
        self.tracker = ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_bbox_properties(self, bbox):
        """Helper method to get bbox properties from x1,y1,x2,y2 format."""
        width = bbox.x2 - bbox.x1
        height = bbox.y2 - bbox.y1
        center_x = bbox.x1 + width / 2
        center_y = bbox.y1 + height / 2
        area = width * height
        return {
            "width": width,
            "height": height,
            "center_x": center_x,
            "center_y": center_y,
            "area": area,
            "center": (center_x, center_y),
        }

    def _get_adaptive_confidence_threshold(
        self, detection: Detection, frame_stats: Dict
    ) -> float:
        """Get adaptive confidence threshold based on detection quality and context."""
        if not self.adaptive_thresholds:
            # Use static thresholds if adaptive is disabled
            thresholds = {
                "player": 0.4,
                "ball": 0.2,
                "referee": 0.35,
                "goalkeeper": 0.35,
            }
            obj_type = detection.object_type or "player"
            return thresholds.get(obj_type, 0.3)

        base_threshold = {
            "player": 0.3,
            "ball": 0.15,
            "referee": 0.25,
            "goalkeeper": 0.25,
        }
        obj_type = detection.object_type or "player"
        base_threshold_val = base_threshold.get(obj_type, 0.25)

        # Adjust based on detection density
        detection_density = frame_stats.get("detection_density", 0)
        if detection_density > 0.02:  # High density - be more selective
            base_threshold_val += 0.1
        elif detection_density < 0.005:  # Low density - be more permissive
            base_threshold_val -= 0.05

        # Adjust based on detection size
        size_ratio = frame_stats.get("size_ratio", 0.01)
        if size_ratio < 0.001:  # Very small detection
            base_threshold_val += 0.1
        elif size_ratio > 0.05:  # Very large detection
            base_threshold_val += 0.05

        return max(0.1, min(0.8, base_threshold_val))

    def _validate_detection_size(self, detection: Detection) -> bool:
        """Validate detection size relative to frame."""
        if not self.size_validation or self.frame_dimensions is None:
            return True

        bbox_props = self._get_bbox_properties(detection.bbox)
        detection_area = bbox_props["area"]
        frame_area = self.frame_dimensions[0] * self.frame_dimensions[1]
        size_ratio = detection_area / frame_area

        return self.min_size_threshold <= size_ratio <= self.max_size_threshold

    def _validate_detection_position(self, detection: Detection) -> bool:
        """Validate detection is within reasonable field boundaries."""
        if not self.spatial_validation or self.frame_dimensions is None:
            return True

        bbox_props = self._get_bbox_properties(detection.bbox)
        frame_h, frame_w = self.frame_dimensions

        # Check if detection is within frame boundaries with some margin
        margin = 0.05  # 5% margin
        min_x, min_y = frame_w * margin, frame_h * margin
        max_x, max_y = frame_w * (1 - margin), frame_h * (1 - margin)

        center_x, center_y = bbox_props["center"]

        return min_x <= center_x <= max_x and min_y <= center_y <= max_y

    def _validate_detection_speed(self, detection: Detection, track_id: int) -> bool:
        """Validate detection movement speed is realistic."""
        if not self.temporal_validation or track_id not in self.track_history:
            return True

        history = self.track_history[track_id]
        if len(history) < 2:
            return True

        # Get last position
        last_pos = history[-1]
        bbox_props = self._get_bbox_properties(detection.bbox)
        current_center = bbox_props["center"]
        last_center = last_pos["center"]

        # Calculate pixel distance
        pixel_distance = np.sqrt(
            (current_center[0] - last_center[0]) ** 2
            + (current_center[1] - last_center[1]) ** 2
        )

        # Frame difference
        frame_diff = self.frame_count - last_pos["frame"]
        if frame_diff == 0:
            return True

        # Convert to real-world speed (assuming 1 pixel ≈ 0.1 meter for football field)
        # This is a rough approximation - in practice, you'd use field transformation
        pixel_to_meter = 0.1
        distance_meters = pixel_distance * pixel_to_meter
        time_seconds = frame_diff / 30.0  # Assuming 30 FPS
        speed_ms = distance_meters / time_seconds if time_seconds > 0 else 0

        return speed_ms <= self.max_speed_threshold

    def _update_track_history(self, detection: Detection, track_id: int):
        """Update track history for temporal validation."""
        if track_id == -1:
            return

        if track_id not in self.track_history:
            self.track_history[track_id] = []

        bbox_props = self._get_bbox_properties(detection.bbox)
        center = bbox_props["center"]

        entry = {
            "frame": self.frame_count,
            "center": center,
            "confidence": detection.confidence,
            "object_type": detection.object_type,
        }

        self.track_history[track_id].append(entry)

        # Keep only recent history
        max_history = max(self.stability_window * 2, 20)
        if len(self.track_history[track_id]) > max_history:
            self.track_history[track_id] = self.track_history[track_id][-max_history:]

    def _is_track_stable(self, track_id: int) -> bool:
        """Check if a track is stable based on recent history."""
        if track_id not in self.track_history:
            return False

        history = self.track_history[track_id]
        if len(history) < self.stability_window:
            return True  # Not enough history to judge instability

        recent_history = history[-self.stability_window :]

        # Check confidence stability
        confidences = [h["confidence"] for h in recent_history]
        conf_std = np.std(confidences)
        if conf_std > 0.2:  # High confidence variance
            return False

        # Check position stability (movement should be smooth)
        positions = [h["center"] for h in recent_history]
        movements = []
        for i in range(1, len(positions)):
            movement = np.sqrt(
                (positions[i][0] - positions[i - 1][0]) ** 2
                + (positions[i][1] - positions[i - 1][1]) ** 2
            )
            movements.append(movement)

        if movements and np.std(movements) > np.mean(movements) * 2:  # Erratic movement
            return False

        return True

    def _filter_detections_advanced(
        self, detections: List[Detection]
    ) -> List[Detection]:
        """Apply advanced filtering logic to detections."""
        if not self.enable_advanced_filtering:
            # Fall back to basic filtering
            return self._filter_detections_basic(detections)

        filtered_detections = []

        # Calculate frame statistics for adaptive thresholds
        if self.frame_dimensions is None and detections:
            # Estimate frame dimensions from detections (rough approximation)
            max_x = max(det.bbox.x2 for det in detections)
            max_y = max(det.bbox.y2 for det in detections)
            self.frame_dimensions = (max_y, max_x)

        total_detection_area = sum(
            self._get_bbox_properties(det.bbox)["area"] for det in detections
        )
        frame_area = (
            self.frame_dimensions[0] * self.frame_dimensions[1]
            if self.frame_dimensions
            else 1
        )

        frame_stats = {
            "detection_density": total_detection_area / frame_area,
            "detection_count": len(detections),
        }

        for detection in detections:
            # Size validation
            if not self._validate_detection_size(detection):
                continue

            # Position validation
            if not self._validate_detection_position(detection):
                continue

            # Calculate frame-specific stats for this detection
            bbox_props = self._get_bbox_properties(detection.bbox)
            size_ratio = bbox_props["area"] / frame_area
            frame_stats["size_ratio"] = size_ratio

            # Adaptive confidence threshold
            min_conf = self._get_adaptive_confidence_threshold(detection, frame_stats)

            if detection.confidence >= min_conf:
                filtered_detections.append(detection)

        return filtered_detections

    def _filter_detections_basic(self, detections: List[Detection]) -> List[Detection]:
        """Apply basic filtering logic (original implementation)."""
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

        return filtered_detections

    def process(self, data: VideoData) -> VideoData:
        """Process video data with enhanced tracking capabilities."""
        # Use progress bar for tracking
        progress_bar = tqdm(data.frames, desc="Enhanced object tracking", unit="frames")

        # Reset tracking state
        self.frame_count = 0
        self.track_history.clear()

        for frame_data in progress_bar:
            detections = frame_data.detections or []
            self.frame_count += 1

            if not detections:
                continue

            # Apply advanced or basic filtering
            if self.enable_advanced_filtering:
                filtered_detections = self._filter_detections_advanced(detections)
            else:
                filtered_detections = self._filter_detections_basic(detections)

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

            # Assign track IDs back to filtered detections with additional validation
            for detection, tid in zip(filtered_detections, track_ids):
                if detection.metadata is None:
                    detection.metadata = {}

                final_track_id = int(tid) if tid is not None else -1

                # Additional temporal validation for existing tracks
                if (
                    final_track_id != -1
                    and self.temporal_validation
                    and not self._validate_detection_speed(detection, final_track_id)
                ):
                    final_track_id = -1  # Reject implausible movement

                detection.metadata["track_id"] = final_track_id

                # Update track history
                self._update_track_history(detection, final_track_id)

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

        # Log enhancement statistics
        if self.enable_advanced_filtering:
            total_tracks = len(self.track_history)
            stable_tracks = sum(
                1 for tid in self.track_history.keys() if self._is_track_stable(tid)
            )
            self.logger.info(
                f"Enhanced tracking completed: {total_tracks} tracks, "
                f"{stable_tracks} stable tracks "
                f"({stable_tracks/max(total_tracks,1)*100:.1f}% stability)"
            )

        return data
