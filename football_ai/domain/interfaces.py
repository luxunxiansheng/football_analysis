"""
Core interfaces for the modern football analysis system (modular pipeline only).
"""

from abc import ABC, abstractmethod
from typing import Any
from .data_models import VideoData


class Processor(ABC):
    @abstractmethod
    def process(self, data: VideoData) -> VideoData:
        """
        Process the input VideoData and return the result.
        """
        pass
