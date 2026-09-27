"""
Validation functions for the HealthConnect ML pipeline (Week 8).
Ensures data integrity, strict schema validation, input/output validation,
and threshold constraints with clear error messages.
"""

from typing import Union, List, Dict
import pandas as pd
import numpy as np

from pipeline.config import (
    REQUIRED_RAW_COLS,
    PREDICTION_INPUT_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    logger,
)


class ValidationError(Exception):
    """Custom exception raised when pipeline validation fails."""
    pass


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate that raw data contains all required columns and is non-empty."""
    if df is None or df.empty:
        raise ValidationError("Raw dataset is empty or None.")

    missing = REQUIRED_RAW_COLS - set(df.columns)
    if missing:
        raise ValidationError(f"Raw dataset is missing required columns: {sorted(missing)}")

    logger.info("Raw data validation passed: %d rows, %d columns.", len(df), len(df.columns))


def validate_clean_data(df: pd.DataFrame) -> None:
    """Validate cleaned data: leakage columns dropped, target is binary, no nulls."""
    if df is None or df.empty:
        raise ValidationError("Cleaned dataset is empty or None.")

    # Leakage check: post-arrival waiting_time_minutes must NOT exist
    if "waiting_time_minutes" in df.columns:
        raise ValidationError("Data leakage detected: 'waiting_time_minutes' present in cleaned dataset.")

    if "target" not in df.columns:
        raise ValidationError("Cleaned dataset missing binary 'target' column.")

    unique_targets = set(df["target"].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValidationError(f"Target column must be binary {0, 1}, found {unique_targets}")

    # Check for remaining nulls
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise ValidationError(f"Cleaned dataset has null values in columns: {cols_with_nulls.to_dict()}")

    logger.info("Cleaned data validation passed: %d rows, target balance: %.3f.", len(df), df["target"].mean())


def validate_features(df: pd.DataFrame, expected_features: List[str]) -> None:
    """Validate that engineered features exist in the DataFrame."""
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        raise ValidationError(f"Engineered dataset is missing expected features: {missing}")

    logger.info("Feature engineering validation passed: all expected features present.")


def validate_feature_names(preprocessor) -> None:
    """Validate that preprocessor produces feature names without errors."""
    try:
        names = preprocessor.get_feature_names_out()
        if len(names) == 0:
            raise ValidationError("Preprocessor generated 0 feature names.")
        logger.info("Preprocessor feature names verified: %d transformed features.", len(names))
    except Exception as e:
        raise ValidationError(f"Failed to extract feature names from preprocessor: {e}")


def validate_threshold(threshold: float) -> None:
    """Validate that the decision threshold is a float in (0, 1)."""
    if not isinstance(threshold, (int, float)):
        raise ValidationError(f"Threshold must be numeric, got {type(threshold).__name__}: {threshold}")
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValidationError(f"Threshold must be strictly within (0, 1), got {threshold}")


def validate_prediction_input(
    data: Union[dict, pd.DataFrame],
    required_cols: List[str] = PREDICTION_INPUT_COLS,
) -> None:
    """Validate inference input against the agreed schema contract."""
    if isinstance(data, dict):
        missing = [col for col in required_cols if col not in data]
        if missing:
            raise ValidationError(f"Prediction input record is missing required fields: {missing}")
        # Check for nulls/None
        null_keys = [k for k, v in data.items() if k in required_cols and (v is None or (isinstance(v, float) and np.isnan(v)))]
        if null_keys:
            raise ValidationError(f"Prediction input contains null/None values for keys: {null_keys}")
        # Prevent data leakage
        if "waiting_time_minutes" in data:
            raise ValidationError("Data leakage alert: 'waiting_time_minutes' provided in inference input.")

    elif isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValidationError("Prediction input DataFrame is empty.")
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValidationError(f"Prediction input DataFrame is missing columns: {missing}")
        null_counts = data[required_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            raise ValidationError(f"Prediction DataFrame contains null values: {cols_with_nulls.to_dict()}")
        if "waiting_time_minutes" in data.columns:
            raise ValidationError("Data leakage alert: 'waiting_time_minutes' column present in batch inference.")
    else:
        raise ValidationError(f"Prediction input must be dict or DataFrame, got {type(data).__name__}")


def validate_prediction_output(preds: pd.DataFrame) -> None:
    """Validate the prediction output schema."""
    required = {"appointment_id", "no_show_probability", "risk_tier"}
    missing = required - set(preds.columns)
    if missing:
        raise ValidationError(f"Prediction output missing required columns: {missing}")

    probs = preds["no_show_probability"]
    if not ((probs >= 0.0) & (probs <= 1.0)).all():
        raise ValidationError("Prediction probabilities must fall within [0.0, 1.0]")

    valid_tiers = {"HIGH_RISK", "MODERATE_RISK", "LOW_RISK"}
    invalid_tiers = set(preds["risk_tier"]) - valid_tiers
    if invalid_tiers:
        raise ValidationError(f"Invalid risk tiers found: {invalid_tiers}")

    logger.info("Prediction output schema successfully validated.")
