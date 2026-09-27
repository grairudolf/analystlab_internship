"""
Data processing and preprocessing pipeline for HealthConnect Week 8.
Handles data loading, cleaning, missingness imputation, leakage removal,
patient-level grouped train/test splitting, and scikit-learn ColumnTransformer.
"""

import os
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from pipeline.config import (
    CATEGORICAL_COLS,
    DATA_PATH,
    DROP_COLS,
    ENGINEERED_CAT_COLS,
    ENGINEERED_NUM_COLS,
    GROUP_COL,
    NEGATIVE_CLASS,
    NUMERICAL_COLS,
    POSITIVE_CLASS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    logger,
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw HealthConnect appointment data from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"HealthConnect data file not found at: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %d rows, %d columns from %s", len(df), len(df.columns), path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw appointment dataset:
    - Filter to binary target (Attended vs No-Show)
    - Encode target column (1 = No-Show, 0 = Attended)
    - Impute reminder_channel missing values as 'None_Sent' (MNAR)
    - Impute distance_to_clinic_km missing values with median (MCAR)
    - Drop post-arrival leakage feature 'waiting_time_minutes'
    - Retain patient_id and appointment_id for grouped split & tracking
    """
    df = df.copy()

    # Target filtering and encoding
    if TARGET_COL in df.columns:
        df = df[df[TARGET_COL].isin([POSITIVE_CLASS, NEGATIVE_CLASS])].reset_index(drop=True)
        df["target"] = (df[TARGET_COL] == POSITIVE_CLASS).astype(int)
        df.drop(columns=[TARGET_COL], inplace=True)

    # Impute missing reminder_channel (NMAR)
    if "reminder_channel" in df.columns:
        df["reminder_channel"] = df["reminder_channel"].fillna("None_Sent")

    # Impute missing distance_to_clinic_km with median (MCAR)
    if "distance_to_clinic_km" in df.columns:
        median_dist = df["distance_to_clinic_km"].median()
        df["distance_to_clinic_km"] = df["distance_to_clinic_km"].fillna(median_dist)

    # Standardize string categories
    cat_candidates = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in cat_candidates:
        df[col] = df[col].astype(str).str.strip()

    # Drop explicit post-arrival leakage: waiting_time_minutes occurs after patient arrival
    if "waiting_time_minutes" in df.columns:
        df.drop(columns=["waiting_time_minutes"], inplace=True)
        logger.info("Dropped post-arrival leakage feature 'waiting_time_minutes'.")

    # Drop non-predictive date strings
    for date_col in ["booking_date", "appointment_date"]:
        if date_col in df.columns:
            df.drop(columns=[date_col], inplace=True)

    logger.info("Data cleaned: %d rows remaining, target mean: %.3f", len(df), df["target"].mean() if "target" in df.columns else 0.0)
    return df


def patient_level_split(
    df: pd.DataFrame,
    group_col: str = GROUP_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a patient-level grouped train/test split (GroupShuffleSplit).
    Guarantees zero patient leakage: all appointments for a given patient
    belong strictly to either the training set or the test set.
    """
    if group_col not in df.columns:
        logger.warning("Group column '%s' not found. Falling back to stratified split.", group_col)
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=test_size, stratify=df.get("target"), random_state=random_state)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Verify zero patient overlap
    overlap = set(train_df[group_col]).intersection(set(test_df[group_col]))
    assert len(overlap) == 0, f"Patient leakage detected: {len(overlap)} patients overlap!"

    logger.info(
        "Patient-level grouped split complete: Train=%d appointments (%d unique patients), Test=%d appointments (%d unique patients).",
        len(train_df), train_df[group_col].nunique(),
        len(test_df), test_df[group_col].nunique(),
    )
    return train_df, test_df


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Construct scikit-learn ColumnTransformer for numerical and categorical features.
    Configured with handle_unknown='ignore' to guarantee inference stability on raw strings.
    """
    all_num_cols = NUMERICAL_COLS + ENGINEERED_NUM_COLS
    all_cat_cols = CATEGORICAL_COLS + ENGINEERED_CAT_COLS

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, all_num_cols),
            ("cat", categorical_pipeline, all_cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract output feature names from the fitted ColumnTransformer."""
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            encoder = trans.named_steps["onehot"]
            names.extend(encoder.get_feature_names_out(cols).tolist())
    return names
