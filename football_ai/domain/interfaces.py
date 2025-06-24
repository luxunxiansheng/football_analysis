"""
Core interfaces for the modern football analysis system (modular pipeline only).
"""

from abc import ABC, abstractmethod
from typing import Any
from .video import Video


class Processor(ABC):
    @abstractmethod
    def process(self, data: Video) -> Video:
        """
        Process the input VideoData and return the result.
        """
        pass
