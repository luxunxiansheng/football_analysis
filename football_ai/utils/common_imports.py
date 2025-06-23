"""
Common imports used across the football analysis system.
"""

# Standard library imports
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union, Set
from collections import defaultdict

# Third-party imports
import numpy as np
import cv2
from tqdm import tqdm
import torch

# Common exceptions and warnings
import warnings

# Suppress common warnings
warnings.filterwarnings(
    "ignore", message=".*force_all_finite.*", category=FutureWarning
)

__all__ = [
    # Standard library
    "os",
    "logging",
    "Path",
    "defaultdict",
    # Typing
    "List",
    "Dict",
    "Optional",
    "Any",
    "Tuple",
    "Union",
    "Set",
    # Third party
    "np",
    "cv2",
    "tqdm",
    "torch",
    "warnings",
]
