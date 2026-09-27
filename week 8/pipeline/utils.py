"""
Utility helper functions for HealthConnect Week 8.
"""

import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy and random."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def format_currency(value: float) -> str:
    """Format numeric value as USD currency."""
    return f"${value:,.2f}"
