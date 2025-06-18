"""
Track ID Optimizer - Post-processes track IDs to reduce fragmentation and improve continuity.
"""

from typing import Dict, List, Set, Any
from collections import defaultdict
import sys
import os

# Add the parent directory to sys.path to import numpy
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    import numpy as np
except ImportError:
    # If numpy is not available, use basic math
    class NumpyStub:
        def sqrt(self, x):
            return x**0.5

    np = NumpyStub()

from ..domain.data_models import VideoData


class TrackIDOptimizer:
    """
    Optimizes track IDs by consolidating fragmented tracks and renumbering sequentially.

    This class addresses common tracking issues:
    1. Fragmented tracks (same object gets multiple IDs)
    2. Large gaps in track ID numbers
    3. Very short-lived tracks that should be merged
    """

    def __init__(
        self,
        min_track_length: int = 5,
        max_merge_distance: float = 50.0,
        max_merge_frames: int = 10,
    ):
        """
        Initialize the track ID optimizer.

        Args:
            min_track_length: Minimum frames for a track to be considered valid
            max_merge_distance: Maximum distance (pixels) to consider merging tracks
            max_merge_frames: Maximum frame gap to consider merging tracks
        """
        self.min_track_length = min_track_length
        self.max_merge_distance = max_merge_distance
        self.max_merge_frames = max_merge_frames

    def optimize_tracks(self, video_data: VideoData) -> VideoData:
        """
        Optimize track IDs in the video data.

        Args:
            video_data: Video data with track IDs to optimize

        Returns:
            Video data with optimized track IDs
        """
        # Step 1: Collect track information
        track_info = self._collect_track_info(video_data)

        # Step 2: Remove very short tracks
        valid_tracks = self._filter_short_tracks(track_info)

        # Step 3: Merge fragmented tracks
        merged_tracks = self._merge_fragmented_tracks(valid_tracks, video_data)

        # Step 4: Renumber tracks sequentially
        track_mapping = self._create_sequential_mapping(merged_tracks)

        # Step 5: Apply the new track IDs
        self._apply_track_mapping(video_data, track_mapping)

        return video_data

    def _collect_track_info(self, video_data: VideoData) -> Dict[int, Dict[str, Any]]:
        """Collect information about all tracks."""
        track_info: Dict[int, Dict[str, Any]] = {}

        for frame_num, frame in enumerate(video_data.frames):
            if not frame.detections:
                continue

            for detection in frame.detections:
                if hasattr(detection, "metadata") and detection.metadata:
                    track_id = detection.metadata.get("track_id")
                    if track_id is not None and track_id > 0:
                        if track_id not in track_info:
                            track_info[track_id] = {
                                "frames": [],
                                "positions": [],
                                "object_type": "",
                                "confidences": [],
                            }

                        info = track_info[track_id]
                        info["frames"].append(frame_num)

                        # Get position from bbox center
                        bbox = detection.bbox
                        center_x = (bbox.x1 + bbox.x2) / 2
                        center_y = (bbox.y1 + bbox.y2) / 2
                        info["positions"].append((center_x, center_y))
                        info["confidences"].append(detection.confidence)

                        if info["object_type"] == "":
                            info["object_type"] = detection.object_type or "unknown"

        return track_info

    def _filter_short_tracks(
        self, track_info: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """Remove tracks that are too short to be meaningful."""
        valid_tracks = {}

        for track_id, info in track_info.items():
            if len(info["frames"]) >= self.min_track_length:
                valid_tracks[track_id] = info

        return valid_tracks

    def _merge_fragmented_tracks(
        self, track_info: Dict[int, Dict[str, Any]], video_data: VideoData
    ) -> Dict[int, Dict[str, Any]]:
        """Merge tracks that likely belong to the same object."""
        # Group tracks by object type for more targeted merging
        tracks_by_type = defaultdict(list)
        for track_id, info in track_info.items():
            tracks_by_type[info["object_type"]].append((track_id, info))

        merged_tracks = {}
        track_merges = {}  # old_id -> new_id mapping

        for object_type, tracks in tracks_by_type.items():
            # Sort tracks by first appearance
            tracks.sort(key=lambda x: min(x[1]["frames"]))

            for track_id, info in tracks:
                # Check if this track should be merged with an existing one
                merged_into = None

                for existing_id in merged_tracks:
                    if self._should_merge_tracks(info, merged_tracks[existing_id]):
                        merged_into = existing_id
                        break

                if merged_into:
                    # Merge this track into existing one
                    existing_info = merged_tracks[merged_into]
                    existing_info["frames"].extend(info["frames"])
                    existing_info["positions"].extend(info["positions"])
                    existing_info["confidences"].extend(info["confidences"])
                    track_merges[track_id] = merged_into
                else:
                    # This is a new unique track
                    merged_tracks[track_id] = info

        return merged_tracks

    def _should_merge_tracks(
        self, track1: Dict[str, Any], track2: Dict[str, Any]
    ) -> bool:
        """Determine if two tracks should be merged."""
        # Don't merge different object types
        if track1["object_type"] != track2["object_type"]:
            return False

        frames1 = set(track1["frames"])
        frames2 = set(track2["frames"])

        # Don't merge overlapping tracks
        if frames1 & frames2:
            return False

        # Check temporal proximity
        max_frame1 = max(frames1)
        min_frame2 = min(frames2)

        frame_gap = abs(min_frame2 - max_frame1)
        if frame_gap > self.max_merge_frames:
            return False

        # Check spatial proximity at the boundary
        # Get last position of track1 and first position of track2
        last_pos1 = track1["positions"][-1] if track1["positions"] else None
        first_pos2 = track2["positions"][0] if track2["positions"] else None

        if last_pos1 and first_pos2:
            distance = (
                (last_pos1[0] - first_pos2[0]) ** 2
                + (last_pos1[1] - first_pos2[1]) ** 2
            ) ** 0.5
            if distance > self.max_merge_distance:
                return False

        return True

    def _create_sequential_mapping(
        self, merged_tracks: Dict[int, Dict[str, Any]]
    ) -> Dict[int, int]:
        """Create a mapping from old track IDs to new sequential IDs."""
        # Sort tracks by object type and first appearance for consistent numbering
        track_list = []
        for track_id, info in merged_tracks.items():
            first_frame = min(info["frames"])
            track_list.append((track_id, info["object_type"], first_frame))

        # Sort: players first, then others, by appearance time
        track_list.sort(
            key=lambda x: (
                (
                    0
                    if x[1] == "player"
                    else 1 if x[1] == "goalkeeper" else 2 if x[1] == "referee" else 3
                ),
                x[2],  # then by first appearance
            )
        )

        # Create sequential mapping
        mapping = {}
        new_id = 1
        for old_id, _, _ in track_list:
            mapping[old_id] = new_id
            new_id += 1

        return mapping

    def _apply_track_mapping(
        self, video_data: VideoData, track_mapping: Dict[int, int]
    ):
        """Apply the new track ID mapping to all detections."""
        for frame in video_data.frames:
            if not frame.detections:
                continue

            for detection in frame.detections:
                if hasattr(detection, "metadata") and detection.metadata:
                    old_track_id = detection.metadata.get("track_id")
                    if old_track_id in track_mapping:
                        detection.metadata["track_id"] = track_mapping[old_track_id]
                    elif old_track_id is not None and old_track_id > 0:
                        # Track was filtered out, mark as untracked
                        detection.metadata["track_id"] = -1
