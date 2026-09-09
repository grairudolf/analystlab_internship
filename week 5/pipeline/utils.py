"""
Shared utility helpers.
"""

import os
import random

import numpy as np

from pipeline.config import RANDOM_STATE


def set_global_seed(seed: int = RANDOM_STATE):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
