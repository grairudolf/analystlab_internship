"""
Reliability and robustness tests for HealthConnect ML pipeline (Week 8).
"""

import os
import shutil
import pytest
import pandas as pd
from pipeline.model_integration import HealthConnectPipeline
from pipeline.validation import ValidationError


@pytest.fixture(scope="module")
def fitted_pipeline():
    pipe = HealthConnectPipeline(threshold=0.35)
    pipe.run(model_name="Random Forest (Final DS Spec)")
    return pipe


def test_invalid_threshold_boundary_low():
    with pytest.raises(ValidationError, match="within"):
        HealthConnectPipeline(threshold=-0.05)


def test_invalid_threshold_boundary_high():
    with pytest.raises(ValidationError, match="within"):
        HealthConnectPipeline(threshold=1.05)


def test_invalid_threshold_type():
    with pytest.raises(ValidationError, match="numeric"):
        HealthConnectPipeline(threshold="optimal")


def test_predict_single_missing_required_column(fitted_pipeline):
    record = {
        "gender": "Female",
        "age": 42,
        # missing age_group
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
    with pytest.raises(ValidationError, match="missing required fields"):
        fitted_pipeline.predict_single(record)


def test_predict_single_blocks_data_leakage(fitted_pipeline):
    record = {
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
        "waiting_time_minutes": 35,  # Leakage!
    }
    with pytest.raises(ValidationError, match="Data leakage alert"):
        fitted_pipeline.predict_single(record)


def test_artifact_persistence_roundtrip(fitted_pipeline, tmp_path):
    save_dir = str(tmp_path / "test_artifacts")
    fitted_pipeline.save_artifacts(save_dir)

    loaded = HealthConnectPipeline.load_artifacts(save_dir)
    assert loaded._fitted is True
    assert loaded.model_name == fitted_pipeline.model_name
    assert loaded.threshold == fitted_pipeline.threshold

    record = {
        "appointment_id": "HC-TEST-LOAD",
        "gender": "Male",
        "age": 50,
        "age_group": "45-64",
        "appointment_type": "General Practice",
        "appointment_day": "Monday",
        "appointment_time": "Morning",
        "booking_lead_days": 4,
        "previous_appointments": 5,
        "previous_no_shows": 1,
        "reminder_sent": "Yes",
        "reminder_channel": "WhatsApp",
        "distance_to_clinic_km": 8.0,
    }
    res_orig = fitted_pipeline.predict_single(record)
    res_loaded = loaded.predict_single(record)
    assert res_orig["no_show_probability"] == res_loaded["no_show_probability"]
    assert res_orig["risk_tier"] == res_loaded["risk_tier"]
