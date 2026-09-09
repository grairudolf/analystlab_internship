"""
Input and output validation checks for the HealthConnect ML pipeline.
Week 6: Validates data schemas, model compatibility, and prediction outputs.
"""

import logging
from typing import List, Set

import pandas as pd
import numpy as np

from pipeline.config import (
    REQUIRED_RAW_COLS,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
    POSITIVE_CLASS,
    NEGATIVE_CLASS,
    logger,
)


class ValidationError(Exception):
    """Raised when a pipeline validation check fails."""


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate raw data has all required columns and sufficient rows."""
    missing_cols = REQUIRED_RAW_COLS - set(df.columns)
    if missing_cols:
        raise ValidationError(f"Missing required columns: {missing_cols}")

    if len(df) < 100:
        raise ValidationError(f"Dataset too small: {len(df)} rows (minimum: 100)")

    logger.info("Raw data validation passed: %d rows, %d columns", len(df), len(df.columns))


def validate_clean_data(df: pd.DataFrame) -> None:
    """Validate cleaned data has correct target and no leakage columns."""
    if set(df["target"].unique()) != {0, 1}:
        raise ValidationError(f"Target must be binary (0/1). Found: {df['target'].unique()}")

    leakage_cols = {"waiting_time_minutes", "appointment_id", "patient_id"}
    leaked = leakage_cols & set(df.columns)
    if leaked:
        raise ValidationError(f"Leakage columns still present: {leaked}")

    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0].to_dict()
        raise ValidationError(f"Remaining null values after cleaning: {null_cols}")

    logger.info("Clean data validation passed: %d rows, %d columns", len(df), len(df.columns))


def validate_features(df: pd.DataFrame, expected_engineered: List[str]) -> None:
    """Validate engineered features are present."""
    missing = [c for c in expected_engineered if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing engineered features: {missing}")

    logger.info("Feature validation passed: %d features", len(df.columns))


def validate_prediction_output(predictions: pd.DataFrame) -> None:
    """Validate model prediction output schema and value ranges."""
    required = {"appointment_id", "no_show_probability", "risk_tier"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValidationError(f"Prediction output missing columns: {missing}")

    probs = predictions["no_show_probability"]
    if probs.min() < 0 or probs.max() > 1:
        raise ValidationError(
            f"Probabilities out of range [{probs.min():.4f}, {probs.max():.4f}]"
        )

    valid_tiers = {"HIGH_RISK", "MODERATE_RISK", "LOW_RISK"}
    invalid_tiers = set(predictions["risk_tier"].unique()) - valid_tiers
    if invalid_tiers:
        raise ValidationError(f"Invalid risk tiers: {invalid_tiers}")

    logger.info("Prediction output validation passed: %d predictions", len(predictions))


def validate_feature_names(preprocessor) -> None:
    """Check the fitted preprocessor can extract feature names."""
    try:
        from pipeline.data_processing import get_feature_names
        names = get_feature_names(preprocessor)
        if len(names) == 0:
            raise ValidationError("Feature names list is empty after fitting")
        logger.info("Feature name extraction validated: %d features", len(names))
    except Exception as e:
        raise ValidationError(f"Feature name extraction failed: {e}")
