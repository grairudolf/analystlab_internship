"""
Unit tests for the HealthConnect ML pipeline components.
Week 6: Updated with improved models and validation checks.
"""

import pytest
from pipeline.config import RANDOM_STATE, TEST_SIZE
from pipeline.data_processing import (
    build_preprocessing_pipeline,
    clean_data,
    get_feature_names,
    load_data,
)
from pipeline.feature_engineering import engineer_features
from pipeline.model_training import get_baseline_models, get_improved_models
from pipeline.validation import (
    ValidationError,
    validate_raw_data,
    validate_clean_data,
    validate_features,
)


@pytest.fixture(scope="module")
def raw_data():
    return load_data()


@pytest.fixture(scope="module")
def prepared_data():
    raw = load_data()
    cleaned = clean_data(raw)
    featured = engineer_features(cleaned)
    return featured


# ── Week 5 baseline tests ────────────────────────────────────────────────────

def test_target_is_binary(prepared_data):
    assert set(prepared_data["target"].unique()) == {0, 1}


def test_leakage_columns_dropped(prepared_data):
    for col in ["appointment_id", "patient_id", "waiting_time_minutes", "booking_date"]:
        assert col not in prepared_data.columns


def test_engineered_features_present(prepared_data):
    for col in ["historical_noshow_ratio", "lead_time_bin", "is_new_patient", "has_reminder"]:
        assert col in prepared_data.columns


def test_no_remaining_nulls(prepared_data):
    assert prepared_data.isnull().sum().sum() == 0


def test_preprocessing_shape_consistency(prepared_data):
    pre = build_preprocessing_pipeline()
    X = prepared_data.drop(columns=["target"])
    Xtr = pre.fit_transform(X)
    Xte = pre.transform(X.head(10))
    assert Xtr.shape[1] == Xte.shape[1]
    assert len(get_feature_names(pre)) == Xtr.shape[1]


def test_baseline_models_train_and_predict(prepared_data):
    pre = build_preprocessing_pipeline()
    X = prepared_data.drop(columns=["target"])
    y = prepared_data["target"]
    Xtr = pre.fit_transform(X)
    for name, model in get_baseline_models().items():
        model.fit(Xtr, y)
        preds = model.predict(Xtr[:5])
        assert len(preds) == 5


# ── Week 6 validation tests ──────────────────────────────────────────────────

def test_validate_raw_data_passes(raw_data):
    validate_raw_data(raw_data)


def test_validate_raw_data_fails_missing_col(raw_data):
    df = raw_data.drop(columns=["gender"])
    with pytest.raises(ValidationError, match="Missing required columns"):
        validate_raw_data(df)


def test_validate_clean_data_passes(prepared_data):
    validate_clean_data(prepared_data)


def test_validate_features_passes(prepared_data):
    expected = ["historical_noshow_ratio", "lead_time_bin", "is_new_patient", "has_reminder"]
    validate_features(prepared_data, expected)


def test_validate_features_fails(prepared_data):
    with pytest.raises(ValidationError, match="Missing engineered features"):
        validate_features(prepared_data, ["nonexistent_feature"])


# ── Week 6 improved models ───────────────────────────────────────────────────

def test_improved_models_train_and_predict(prepared_data):
    pre = build_preprocessing_pipeline()
    X = prepared_data.drop(columns=["target"])
    y = prepared_data["target"]
    Xtr = pre.fit_transform(X)
    for name, model in get_improved_models().items():
        model.fit(Xtr, y)
        preds = model.predict(Xtr[:5])
        assert len(preds) == 5
