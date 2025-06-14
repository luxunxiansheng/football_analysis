"""
Bounding Box Utilities Module

This module provides utility functions for bounding box operations
including calculations, transformations, and format conversions.
"""

import numpy as np
from typing import List, Tuple, Union, Optional


def xyxy_to_xywh(
    bbox: Union[List[float], Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from (x1, y1, x2, y2) to (x, y, w, h) format.

    Args:
        bbox: Bounding box in (x1, y1, x2, y2) format

    Returns:
        Bounding box in (x, y, w, h) format
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return (x1, y1, w, h)


def xywh_to_xyxy(
    bbox: Union[List[float], Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from (x, y, w, h) to (x1, y1, x2, y2) format.

    Args:
        bbox: Bounding box in (x, y, w, h) format

    Returns:
        Bounding box in (x1, y1, x2, y2) format
    """
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h
    return (x, y, x2, y2)


def calculate_bbox_center(
    bbox: Union[List[float], Tuple[float, float, float, float]],
) -> Tuple[float, float]:
    """
    Calculate the center point of a bounding box.

    Args:
        bbox: Bounding box in (x1, y1, x2, y2) format

    Returns:
        Center point (x, y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)


def calculate_bbox_area(
    bbox: Union[List[float], Tuple[float, float, float, float]],
) -> float:
    """
    Calculate the area of a bounding box.

    Args:
        bbox: Bounding box in (x1, y1, x2, y2) format

    Returns:
        Area of the bounding box
    """
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height


def calculate_iou(
    bbox1: Union[List[float], Tuple[float, float, float, float]],
    bbox2: Union[List[float], Tuple[float, float, float, float]],
) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bounding box in (x1, y1, x2, y2) format
        bbox2: Second bounding box in (x1, y1, x2, y2) format

    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection area
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Check if there's an intersection
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate union area
    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)
    union_area = area1 + area2 - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def calculate_distance_between_centers(
    bbox1: Union[List[float], Tuple[float, float, float, float]],
    bbox2: Union[List[float], Tuple[float, float, float, float]],
) -> float:
    """
    Calculate Euclidean distance between centers of two bounding boxes.

    Args:
        bbox1: First bounding box in (x1, y1, x2, y2) format
        bbox2: Second bounding box in (x1, y1, x2, y2) format

    Returns:
        Distance between centers
    """
    center1 = calculate_bbox_center(bbox1)
    center2 = calculate_bbox_center(bbox2)

    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]

    return float(np.sqrt(dx * dx + dy * dy))


def expand_bbox(
    bbox: Union[List[float], Tuple[float, float, float, float]],
    expand_ratio: float = 0.1,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    Expand a bounding box by a given ratio.

    Args:
        bbox: Bounding box in (x1, y1, x2, y2) format
        expand_ratio: Ratio to expand the bbox (0.1 = 10% expansion)
        frame_width: Width of frame to clamp coordinates
        frame_height: Height of frame to clamp coordinates

    Returns:
        Expanded bounding box
    """
    x1, y1, x2, y2 = bbox

    # Calculate current dimensions
    width = x2 - x1
    height = y2 - y1

    # Calculate expansion amounts
    expand_w = width * expand_ratio / 2
    expand_h = height * expand_ratio / 2

    # Expand the bbox
    new_x1 = x1 - expand_w
    new_y1 = y1 - expand_h
    new_x2 = x2 + expand_w
    new_y2 = y2 + expand_h

    # Clamp to frame boundaries if provided
    if frame_width is not None:
        new_x1 = max(0, new_x1)
        new_x2 = min(frame_width, new_x2)

    if frame_height is not None:
        new_y1 = max(0, new_y1)
        new_y2 = min(frame_height, new_y2)

    return (new_x1, new_y1, new_x2, new_y2)


def crop_bbox_to_frame(
    bbox: Union[List[float], Tuple[float, float, float, float]],
    frame_width: int,
    frame_height: int,
) -> Tuple[float, float, float, float]:
    """
    Crop bounding box coordinates to fit within frame boundaries.

    Args:
        bbox: Bounding box in (x1, y1, x2, y2) format
        frame_width: Width of the frame
        frame_height: Height of the frame

    Returns:
        Cropped bounding box
    """
    x1, y1, x2, y2 = bbox

    # Clamp coordinates to frame boundaries
    x1 = max(0, min(x1, frame_width))
    y1 = max(0, min(y1, frame_height))
    x2 = max(x1, min(x2, frame_width))
    y2 = max(y1, min(y2, frame_height))

    return (x1, y1, x2, y2)


def get_foot_position(
    bbox: Union[List[float], Tuple[float, float, float, float]],
) -> Tuple[float, float]:
    """
    Get the foot position (bottom center) of a bounding box.
    Useful for player tracking where foot position is more stable.

    Args:
        bbox: Bounding box in (x1, y1, x2, y2) format

    Returns:
        Foot position (x, y)
    """
    x1, y1, x2, y2 = bbox
    foot_x = (x1 + x2) / 2
    foot_y = y2
    return (foot_x, foot_y)


def scale_bbox(
    bbox: Union[List[float], Tuple[float, float, float, float]],
    scale_x: float,
    scale_y: float,
) -> Tuple[float, float, float, float]:
    """
    Scale a bounding box by given factors.

    Args:
        bbox: Bounding box in (x1, y1, x2, y2) format
        scale_x: Scaling factor for x coordinates
        scale_y: Scaling factor for y coordinates

    Returns:
        Scaled bounding box
    """
    x1, y1, x2, y2 = bbox

    new_x1 = x1 * scale_x
    new_y1 = y1 * scale_y
    new_x2 = x2 * scale_x
    new_y2 = y2 * scale_y

    return (new_x1, new_y1, new_x2, new_y2)


def filter_bboxes_by_size(
    bboxes: List[Tuple[float, float, float, float]],
    min_area: float = 0,
    max_area: float = float("inf"),
    min_width: float = 0,
    min_height: float = 0,
) -> List[Tuple[float, float, float, float]]:
    """
    Filter bounding boxes by size criteria.

    Args:
        bboxes: List of bounding boxes in (x1, y1, x2, y2) format
        min_area: Minimum area threshold
        max_area: Maximum area threshold
        min_width: Minimum width threshold
        min_height: Minimum height threshold

    Returns:
        Filtered list of bounding boxes
    """
    filtered_bboxes = []

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height

        # Apply filters
        if (
            area >= min_area
            and area <= max_area
            and width >= min_width
            and height >= min_height
        ):
            filtered_bboxes.append(bbox)

    return filtered_bboxes


def merge_overlapping_bboxes(
    bboxes: List[Tuple[float, float, float, float]], iou_threshold: float = 0.5
) -> List[Tuple[float, float, float, float]]:
    """
    Merge overlapping bounding boxes based on IoU threshold.

    Args:
        bboxes: List of bounding boxes in (x1, y1, x2, y2) format
        iou_threshold: IoU threshold for merging

    Returns:
        List of merged bounding boxes
    """
    if not bboxes:
        return []

    merged_bboxes = []
    used = [False] * len(bboxes)

    for i, bbox1 in enumerate(bboxes):
        if used[i]:
            continue

        # Start with current bbox
        merged_bbox = list(bbox1)
        used[i] = True

        # Check for overlapping bboxes
        for j, bbox2 in enumerate(bboxes):
            if i == j or used[j]:
                continue

            iou = calculate_iou(bbox1, bbox2)
            if iou >= iou_threshold:
                # Merge bboxes by taking min/max coordinates
                merged_bbox[0] = min(merged_bbox[0], bbox2[0])  # x1
                merged_bbox[1] = min(merged_bbox[1], bbox2[1])  # y1
                merged_bbox[2] = max(merged_bbox[2], bbox2[2])  # x2
                merged_bbox[3] = max(merged_bbox[3], bbox2[3])  # y2
                used[j] = True

        merged_bboxes.append(tuple(merged_bbox))

    return merged_bboxes
