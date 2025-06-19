
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.detection.core import Detections
import numpy as np
import logging
from tqdm import tqdm

from ..domain.data_models import VideoData
from ..domain.interfaces import Processor
from .track_optimizer import TrackIDOptimizer


class EnhancedTrackProcessor(Processor):
    """Enhanced TrackProcessor with advanced detection filtering and optimization."""
    
    def __init__(
        self,
        track_activation_threshold: float = 0.4,
        lost_track_buffer: int = 800,
        minimum_matching_threshold: float = 0.75,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 3,
        min_track_length: int = 50,
        max_merge_distance: float = 200.0,
        max_merge_frames: int = 100,
        # Enhanced parameters
        enable_spatial_filtering: bool = True,
        enable_size_filtering: bool = True,
        enable_temporal_validation: bool = True,
        field_boundaries: tuple = (50, 50, 1870, 1030),  # (x1, y1, x2, y2)
    ):
        """Initialize Enhanced TrackProcessor with advanced filtering."""
        
        # Core tracking parameters (optimized)
        self.min_track_length = min_track_length
        self.max_merge_distance = max_merge_distance
        self.max_merge_frames = max_merge_frames
        
        # Enhanced filtering flags
        self.enable_spatial_filtering = enable_spatial_filtering
        self.enable_size_filtering = enable_size_filtering
        self.enable_temporal_validation = enable_temporal_validation
        self.field_boundaries = field_boundaries
        
        # Object-specific confidence thresholds
        self.confidence_thresholds = {
            0: 0.4,   # Players
            1: 0.2,   # Ball
            2: 0.35,  # Referee
            3: 0.35,  # Goalkeeper
        }
        
        # Size constraints (width, height) in pixels
        self.size_constraints = {
            0: (20, 40, 120, 300),   # Players: min_w, min_h, max_w, max_h
            1: (8, 8, 50, 50),       # Ball
            2: (20, 40, 120, 300),   # Referee
            3: (20, 40, 120, 300),   # Goalkeeper
        }
        
        # Temporal validation history
        self.detection_history = {}
        self.validation_window = 5
        
        # Initialize ByteTrack with optimized parameters
        self.tracker = ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        
        # Track optimizer for post-processing
        self.optimizer = TrackIDOptimizer(
            min_track_length=min_track_length,
            max_merge_distance=max_merge_distance,
            max_merge_frames=max_merge_frames,
        )
        
    def filter_detections(self, detections: Detections, frame_idx: int) -> Detections:
        """Apply advanced filtering to detections."""
        if len(detections) == 0:
            return detections
            
        valid_indices = []
        
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            confidence = detections.confidence[i]
            class_id = int(detections.class_id[i])
            
            # 1. Confidence filtering (object-specific)
            min_conf = self.confidence_thresholds.get(class_id, 0.3)
            if confidence < min_conf:
                continue
                
            # 2. Spatial filtering (field boundaries)
            if self.enable_spatial_filtering:
                x1, y1, x2, y2 = bbox
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Check if detection is within field boundaries
                fx1, fy1, fx2, fy2 = self.field_boundaries
                if not (fx1 <= center_x <= fx2 and fy1 <= center_y <= fy2):
                    continue
                    
            # 3. Size filtering
            if self.enable_size_filtering and class_id in self.size_constraints:
                x1, y1, x2, y2 = bbox
                width, height = x2 - x1, y2 - y1
                min_w, min_h, max_w, max_h = self.size_constraints[class_id]
                
                if not (min_w <= width <= max_w and min_h <= height <= max_h):
                    continue
                    
            # 4. Temporal validation
            if self.enable_temporal_validation:
                detection_key = f"{class_id}_{int(center_x//50)}_{int(center_y//50)}"
                
                if detection_key not in self.detection_history:
                    self.detection_history[detection_key] = []
                    
                self.detection_history[detection_key].append(frame_idx)
                
                # Keep only recent history
                self.detection_history[detection_key] = [
                    f for f in self.detection_history[detection_key] 
                    if frame_idx - f <= self.validation_window
                ]
                
                # Require consistency across multiple frames for new detections
                if len(self.detection_history[detection_key]) < 2:
                    continue
                    
            valid_indices.append(i)
            
        # Filter detections to keep only valid ones
        if valid_indices:
            return Detections(
                xyxy=detections.xyxy[valid_indices],
                confidence=detections.confidence[valid_indices],
                class_id=detections.class_id[valid_indices],
                tracker_id=detections.tracker_id[valid_indices] if detections.tracker_id is not None else None
            )
        else:
            return Detections.empty()
    
    def process(self, video_data: VideoData) -> VideoData:
        """Process video data with enhanced tracking."""
        logging.info("Starting enhanced tracking processing...")
        
        processed_frames = []
        total_frames = len(video_data.frames)
        
        with tqdm(total=total_frames, desc="Enhanced Tracking") as pbar:
            for frame_idx, frame in enumerate(video_data.frames):
                # Apply enhanced detection filtering
                filtered_detections = self.filter_detections(frame.detections, frame_idx)
                
                # Update tracker with filtered detections
                tracked_detections = self.tracker.update_with_detections(filtered_detections)
                
                # Create new frame with tracked detections
                processed_frame = frame.model_copy()
                processed_frame.detections = tracked_detections
                processed_frames.append(processed_frame)
                
                pbar.update(1)
                
        # Create processed video data
        processed_video_data = VideoData(frames=processed_frames)
        
        # Apply track optimization
        optimized_video_data = self.optimizer.process(processed_video_data)
        
        logging.info(f"Enhanced tracking completed. Processed {total_frames} frames.")
        return optimized_video_data
