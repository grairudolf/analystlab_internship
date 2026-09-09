"""
Feature engineering pipeline for HealthConnect appointment data.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that strengthen predictive signal for no-show classification.

    Engineered Features:
    1. historical_noshow_ratio — proportion of past appointments missed
    2. lead_time_bin — categorical bucket of booking lead days
    3. is_new_patient — binary flag for patients with no prior appointments
    4. has_reminder — binary simplification of reminder_sent
    """
    df = df.copy()

    # 1. Historical no-show ratio
    df["historical_noshow_ratio"] = df["previous_no_shows"] / df["previous_appointments"].clip(lower=1)

    # 2. Lead time binning
    bins = [0, 3, 7, 14, 30, 60, np.inf]
    labels = ["0-3d", "4-7d", "8-14d", "15-30d", "31-60d", "60d+"]
    df["lead_time_bin"] = pd.cut(
        df["booking_lead_days"], bins=bins, labels=labels, right=True, include_lowest=True
    )

    # 3. New patient flag
    df["is_new_patient"] = (df["previous_appointments"] == 0).astype(int)

    # 4. Binary reminder flag
    df["has_reminder"] = (df["reminder_sent"] == "Yes").astype(int)

    print(f"Feature engineering complete: {df.shape[1]} columns")
    return df
