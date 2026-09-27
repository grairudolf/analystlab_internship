"""
Contract tests for Data Science (Mairame Samba Niang) <-> ML Engineering (Rudolf) handoff.
Validates the technical constraints, input/output schemas, and artifact compatibility
agreed upon during HC-POD collaboration.
"""

import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pipeline.model_integration import HealthConnectPipeline
from pipeline.validation import ValidationError


@pytest.fixture(scope="module")
def fitted_pipeline():
    pipe = HealthConnectPipeline(threshold=0.35)
    pipe.run(model_name="Random Forest (Final DS Spec)")
    return pipe


def test_contract_input_features_accepted(fitted_pipeline):
    """Confirm the exact 12-feature input contract shared with Data Science is accepted."""
    record = {
        "gender": "Female",
        "age": 34,
        "age_group": "25-34",
        "appointment_type": "Specialist Consultation",
        "appointment_day": "Tuesday",
        "appointment_time": "Afternoon",
        "booking_lead_days": 14,
        "previous_appointments": 2,
        "previous_no_shows": 1,
        "reminder_sent": "Yes",
        "reminder_channel": "WhatsApp",
        "distance_to_clinic_km": 11.5,
    }
    out = fitted_pipeline.predict_single(record)
    assert "no_show_probability" in out
    assert "risk_tier" in out
    assert "recommended_action" in out


def test_contract_rejects_leakage_waiting_time(fitted_pipeline):
    """Confirm technical constraint: waiting_time_minutes must NOT be passed."""
    record = {
        "gender": "Male",
        "age": 45,
        "age_group": "45-64",
        "appointment_type": "Follow-up",
        "appointment_day": "Monday",
        "appointment_time": "Morning",
        "booking_lead_days": 5,
        "previous_appointments": 4,
        "previous_no_shows": 0,
        "reminder_sent": "No",
        "reminder_channel": "None_Sent",
        "distance_to_clinic_km": 6.2,
        "waiting_time_minutes": 25,  # Leakage!
    }
    with pytest.raises(ValidationError, match="Data leakage alert"):
        fitted_pipeline.predict_single(record)


def test_contract_output_json_schema(fitted_pipeline):
    """Confirm the exact output JSON structure required by clinic operations."""
    record = {
        "appointment_id": "HC-DS-TEST-999",
        "gender": "Female",
        "age": 22,
        "age_group": "18-24",
        "appointment_type": "General Practice",
        "appointment_day": "Thursday",
        "appointment_time": "Morning",
        "booking_lead_days": 21,
        "previous_appointments": 1,
        "previous_no_shows": 1,
        "reminder_sent": "No",
        "reminder_channel": "None_Sent",
        "distance_to_clinic_km": 28.0,
    }
    res = fitted_pipeline.predict_single(record)
    assert set(res.keys()) == {"appointment_id", "no_show_probability", "risk_tier", "recommended_action"}
    assert isinstance(res["appointment_id"], str)
    assert isinstance(res["no_show_probability"], float)
    assert res["risk_tier"] in {"HIGH_RISK", "MODERATE_RISK", "LOW_RISK"}
    assert isinstance(res["recommended_action"], str)


def test_contract_ds_model_import_adapter(fitted_pipeline):
    """Confirm that an externally trained Random Forest from Data Science can be imported seamlessly."""
    pipe = HealthConnectPipeline(threshold=0.35)
    # Create mock fitted external RF model matching DS specs
    external_rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    # Use internal preprocessor to create mock fit
    dummy_X = fitted_pipeline.preprocessor.transform(
        pipe.REQUIRED_INPUT_COLS and pd.DataFrame([{
            "gender": "Female", "age": 30, "age_group": "25-34",
            "appointment_type": "Follow-up", "appointment_day": "Monday",
            "appointment_time": "Morning", "booking_lead_days": 7,
            "previous_appointments": 3, "previous_no_shows": 0,
            "reminder_sent": "Yes", "reminder_channel": "SMS",
            "distance_to_clinic_km": 5.0,
            "historical_noshow_ratio": 0.0, "lead_time_bin": "Short",
            "is_new_patient": 0, "has_reminder": 1,
        }])
    )
    external_rf.fit(dummy_X, [1])

    # Import DS model
    pipe.preprocessor = fitted_pipeline.preprocessor
    pipe.import_ds_model(external_rf, is_full_pipeline=False, model_name="Mairame_RF_Final")
    assert pipe._fitted is True
    assert pipe.model_name == "Mairame_RF_Final"
