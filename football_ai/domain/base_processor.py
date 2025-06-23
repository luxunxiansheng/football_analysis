"""
Base processor classes with common functionality.
"""

from abc import ABC
from ..utils import logging, setup_logger, Optional
from ..domain.interfaces import Processor


class BaseProcessor(Processor, ABC):
    """
    Base processor class with common functionality like logging.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize base processor.

        Args:
            name: Optional custom name for logging, defaults to class name
        """
        self.name = name or self.__class__.__name__
        self.logger = setup_logger(self.name)

    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)

    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
