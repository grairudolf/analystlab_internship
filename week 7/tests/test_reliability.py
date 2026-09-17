"""
Reliability & error-handling tests for the HealthConnect ML pipeline (Week 7).
Tests invalid/unexpected input scenarios, reproducibility, artifact round-trips,
and threshold validation.
"""

import os

import pandas as pd
import pytest

from pipeline.model_integration import HealthConnectPipeline
from pipeline.validation import ValidationError

# A valid prediction record mirroring the cleaned schema
VALID_RECORD = {
    "gender": "Female",
    "age": 42,
    "age_group": "35-44",
    "appointment_type": "Follow-up",
    "appointment_day": "Tuesday",
    "appointment_time": "Afternoon",
    "booking_lead_days": 18,
    "previous_appointments": 4,
    "previous_no_shows": 2,
    "reminder_sent": "Yes",
    "reminder_channel": "SMS",
    "distance_to_clinic_km": 14.2,
}


@pytest.fixture(scope="module")
def fitted_pipeline():
    pipe = HealthConnectPipeline(threshold=0.50)
    pipe.run()
    return pipe


# ── Threshold validation ─────────────────────────────────────────────────────

def test_invalid_threshold_low():
    with pytest.raises(ValidationError, match="[0, 1]"):
        HealthConnectPipeline(threshold=-0.1)


def test_invalid_threshold_high():
    with pytest.raises(ValidationError, match="[0, 1]"):
        HealthConnectPipeline(threshold=1.1)


def test_invalid_threshold_non_numeric():
    with pytest.raises(ValidationError, match="numeric"):
        HealthConnectPipeline(threshold="high")


def test_valid_threshold_accepted():
    pipe = HealthConnectPipeline(threshold=0.4)
    assert pipe.threshold == 0.4


# ── Prediction input validation ──────────────────────────────────────────────

def test_predict_single_missing_column(fitted_pipeline):
    bad = {k: v for k, v in VALID_RECORD.items() if k != "age"}
    with pytest.raises(ValidationError, match="missing required columns"):
        fitted_pipeline.predict_single(bad)


def test_predict_single_null_value(fitted_pipeline):
    bad = dict(VALID_RECORD)
    bad["age"] = None
    with pytest.raises(ValidationError, match="null value"):
        fitted_pipeline.predict_single(bad)


def test_predict_single_valid_record(fitted_pipeline):
    result = fitted_pipeline.predict_single(VALID_RECORD)
    assert 0 <= result["no_show_probability"] <= 1
    assert result["risk_tier"] in {"HIGH_RISK", "MODERATE_RISK", "LOW_RISK"}


def test_predict_batch_missing_column(fitted_pipeline):
    df = pd.DataFrame([VALID_RECORD])
    df = df.drop(columns=["reminder_channel"])
    with pytest.raises(ValidationError, match="missing required columns"):
        fitted_pipeline.predict_batch(df)


def test_predict_batch_valid(fitted_pipeline):
    df = pd.DataFrame([VALID_RECORD, VALID_RECORD])
    df["appointment_id"] = ["HC-00001", "HC-00002"]
    preds = fitted_pipeline.predict_batch(df)
    assert len(preds) == 2
    assert "risk_tier" in preds.columns


def test_predict_single_before_fit():
    pipe = HealthConnectPipeline()
    with pytest.raises(ValidationError, match="not been fitted"):
        pipe.predict_single(VALID_RECORD)


# ── Artifact save / load round-trip ──────────────────────────────────────────

def test_artifact_roundtrip(fitted_pipeline, tmp_path):
    fitted_pipeline.save_artifacts(str(tmp_path))
    loaded = HealthConnectPipeline.load_artifacts(str(tmp_path))
    assert loaded._fitted is True
    assert loaded.model_name == fitted_pipeline.model_name

    # Same prediction before and after reload
    r1 = fitted_pipeline.predict_single(VALID_RECORD)
    r2 = loaded.predict_single(VALID_RECORD)
    assert r1["no_show_probability"] == r2["no_show_probability"]
    assert r1["risk_tier"] == r2["risk_tier"]


def test_load_artifacts_missing(tmp_path):
    with pytest.raises(ValidationError, match="No pipeline artifacts"):
        HealthConnectPipeline.load_artifacts(str(tmp_path))


def test_save_artifacts_before_fit(tmp_path):
    pipe = HealthConnectPipeline()
    with pytest.raises(ValidationError, match="not been fitted"):
        pipe.save_artifacts(str(tmp_path))


# ── Reproducibility ──────────────────────────────────────────────────────────

def test_reproducibility(fitted_pipeline):
    # Two freshly-run pipelines should select the same best model / feature count
    p1 = HealthConnectPipeline(threshold=0.50)
    r1 = p1.run()
    p2 = HealthConnectPipeline(threshold=0.50)
    r2 = p2.run()
    assert r1["best_model_name"] == r2["best_model_name"]
    assert len(p1.feature_names) == len(p2.feature_names)
