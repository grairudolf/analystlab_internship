"""
Feature engineering module for HealthConnect Week 8.
Constructs domain-specific features without introducing data leakage.
"""

import pandas as pd
import numpy as np
from pipeline.config import logger


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain features from patient appointment history and scheduling context:
    - historical_noshow_ratio: proportion of past missed visits (Laplace smoothed)
    - lead_time_bin: categorized lead time bucket
    - is_new_patient: binary indicator for first-time clinic attendees
    - has_reminder: binary flag for whether an automated reminder was dispatched
    """
    df = df.copy()

    # 1. Historical No-Show Ratio (Laplace smoothed)
    prev_appts = df["previous_appointments"].fillna(0)
    prev_noshows = df["previous_no_shows"].fillna(0)
    df["historical_noshow_ratio"] = prev_noshows / (prev_appts + 1.0)

    # 2. Lead Time Categorization
    lead_days = df["booking_lead_days"].fillna(0)
    df["lead_time_bin"] = pd.cut(
        lead_days,
        bins=[-np.inf, 0, 3, 14, np.inf],
        labels=["Same_Day", "Short", "Medium", "Long"],
    ).astype(str)

    # 3. New Patient Flag
    df["is_new_patient"] = (prev_appts == 0).astype(int)

    # 4. Reminder Sent Binary Flag
    df["has_reminder"] = (df["reminder_sent"] == "Yes").astype(int)

    logger.info("Feature engineering complete: 4 domain features constructed.")
    return df
