"""
Basic smoke tests for the HealthConnect ML pipeline components.
Run with: python -m pytest tests/ -v
"""

import pytest
from sklearn.model_selection import train_test_split

from pipeline.config import RANDOM_STATE, TEST_SIZE
from pipeline.data_processing import (
    build_preprocessing_pipeline,
    clean_data,
    get_feature_names,
    load_data,
)
from pipeline.feature_engineering import engineer_features
from pipeline.model_training import get_baseline_models


@pytest.fixture(scope="module")
def prepared_data():
    raw = load_data()
    cleaned = clean_data(raw)
    featured = engineer_features(cleaned)
    return featured


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


def test_models_train_and_predict(prepared_data):
    pre = build_preprocessing_pipeline()
    X = prepared_data.drop(columns=["target"])
    y = prepared_data["target"]
    Xtr = pre.fit_transform(X)
    for name, model in get_baseline_models().items():
        model.fit(Xtr, y)
        preds = model.predict(Xtr[:5])
        assert len(preds) == 5
