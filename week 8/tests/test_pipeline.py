"""
Unit tests for the HealthConnect ML pipeline components (Week 8).
"""

import pytest
import pandas as pd
import numpy as np
from pipeline.config import DATA_PATH, DROP_COLS
from pipeline.data_processing import (
    load_data,
    clean_data,
    patient_level_split,
    build_preprocessing_pipeline,
    get_feature_names,
)
from pipeline.feature_engineering import engineer_features
from pipeline.validation import (
    validate_raw_data,
    validate_clean_data,
    validate_features,
    ValidationError,
)


def test_load_data_returns_non_empty_df():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) >= 4000


def test_validate_raw_data():
    df = load_data()
    validate_raw_data(df)


def test_clean_data_drops_leakage_waiting_time():
    raw = load_data()
    cleaned = clean_data(raw)
    assert "waiting_time_minutes" not in cleaned.columns
    assert "target" in cleaned.columns
    assert set(cleaned["target"].unique()).issubset({0, 1})


def test_clean_data_imputes_missing_values():
    raw = load_data()
    cleaned = clean_data(raw)
    assert cleaned["reminder_channel"].isnull().sum() == 0
    assert cleaned["distance_to_clinic_km"].isnull().sum() == 0


def test_patient_level_split_has_zero_leakage():
    raw = load_data()
    cleaned = clean_data(raw)
    train_df, test_df = patient_level_split(cleaned, test_size=0.2, random_state=42)
    train_patients = set(train_df["patient_id"])
    test_patients = set(test_df["patient_id"])
    assert len(train_patients.intersection(test_patients)) == 0


def test_feature_engineering_creates_expected_columns():
    raw = load_data()
    cleaned = clean_data(raw)
    featured = engineer_features(cleaned)
    expected = ["historical_noshow_ratio", "lead_time_bin", "is_new_patient", "has_reminder"]
    for col in expected:
        assert col in featured.columns
    validate_features(featured, expected)


def test_preprocessing_pipeline_fit_transform():
    raw = load_data()
    cleaned = clean_data(raw)
    featured = engineer_features(cleaned)
    train_df, test_df = patient_level_split(featured, test_size=0.2, random_state=42)
    preprocessor = build_preprocessing_pipeline()
    X_train = train_df.drop(columns=["target", "patient_id", "appointment_id"])
    X_proc = preprocessor.fit_transform(X_train)
    assert X_proc.shape[0] == len(train_df)
    assert X_proc.shape[1] > 20
    names = get_feature_names(preprocessor)
    assert len(names) == X_proc.shape[1]
