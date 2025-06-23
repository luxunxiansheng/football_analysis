"""
Logging utilities for the football analysis system.
"""

import logging
from typing import Optional


def setup_logger(
    name: str, level: str = "INFO", format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting across the system.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string, uses default if None

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()

        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
