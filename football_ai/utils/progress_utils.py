"""
Progress bar utilities for consistent progress tracking.
"""

from tqdm import tqdm
from typing import Optional, Iterable, Any


def create_progress_bar(
    iterable: Optional[Iterable] = None,
    total: Optional[int] = None,
    desc: str = "Processing",
    unit: str = "items",
    disable: bool = False,
) -> tqdm:
    """
    Create a standardized progress bar.

    Args:
        iterable: Iterable to track progress over
        total: Total number of items (if iterable not provided)
        desc: Description for the progress bar
        unit: Unit name for progress tracking
        disable: Whether to disable the progress bar

    Returns:
        Configured tqdm progress bar
    """
    return tqdm(
        iterable=iterable,
        total=total,
        desc=desc,
        unit=unit,
        disable=disable,
        ncols=80,  # Consistent width
        ascii=True,  # Better compatibility
    )
