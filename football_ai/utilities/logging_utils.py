"""
Logging utilities for the football analysis system.
"""

import logging
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
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

    # Handle invalid level gracefully
    try:
        logger.setLevel(getattr(logging, level.upper()))
    except AttributeError:
        logger.setLevel(logging.INFO)  # Default to INFO if invalid level

    if not logger.handlers:
        # Setup console handler
        handler = logging.StreamHandler()

        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Setup file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
