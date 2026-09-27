"""
Integration tests for the HealthConnect ML pipeline (Week 8).
"""

import pytest
import pandas as pd
from pipeline.model_integration import HealthConnectPipeline
from pipeline.validation import ValidationError, validate_prediction_output


@pytest.fixture(scope="module")
def fitted_pipeline():
    """Fit the full pipeline once for module integration tests."""
    pipe = HealthConnectPipeline(threshold=0.35)
    pipe.run(model_name="Random Forest (Final DS Spec)")
    return pipe


def test_pipeline_fitted_attributes(fitted_pipeline):
    assert fitted_pipeline._fitted is True
    assert fitted_pipeline.model is not None
    assert fitted_pipeline.preprocessor is not None
    assert len(fitted_pipeline.feature_names) > 0
    assert "roc_auc" in fitted_pipeline.evaluation_metrics
    assert fitted_pipeline.evaluation_metrics["roc_auc"] >= 0.65


def test_pipeline_predict_single_returns_contract(fitted_pipeline):
    record = {
        "appointment_id": "HC-INT-001",
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
    result = fitted_pipeline.predict_single(record)
    assert result["appointment_id"] == "HC-INT-001"
    assert 0.0 <= result["no_show_probability"] <= 1.0
    assert result["risk_tier"] in {"HIGH_RISK", "MODERATE_RISK", "LOW_RISK"}
    assert result["recommended_action"] in {
        "TRIGGER_WHATSAPP_CONFIRMATION_AND_BUFFERSLOT",
        "TRIGGER_SMS_REMINDER_REQUESTING_CONFIRMATION",
        "STANDARD_EMAIL_NOTIFICATION",
    }


def test_pipeline_predict_batch_returns_valid_df(fitted_pipeline):
    records = [
        {
            "appointment_id": f"HC-INT-{i:03d}",
            "gender": "Male" if i % 2 == 0 else "Female",
            "age": 30 + i,
            "age_group": "25-34",
            "appointment_type": "General Practice",
            "appointment_day": "Monday",
            "appointment_time": "Morning",
            "booking_lead_days": 5 * i,
            "previous_appointments": i,
            "previous_no_shows": 1 if i > 2 else 0,
            "reminder_sent": "Yes" if i % 2 == 0 else "No",
            "reminder_channel": "SMS" if i % 2 == 0 else "None_Sent",
            "distance_to_clinic_km": 5.0 + i,
        }
        for i in range(5)
    ]
    df = pd.DataFrame(records)
    preds = fitted_pipeline.predict_batch(df)
    assert len(preds) == 5
    validate_prediction_output(preds)


def test_pipeline_raises_before_fit():
    unfitted = HealthConnectPipeline(threshold=0.35)
    with pytest.raises(ValidationError, match="not fitted"):
        unfitted.predict_single({"gender": "Female"})
