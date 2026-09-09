"""
Integration tests for the full HealthConnect ML pipeline (Week 6).
Tests the interaction between data processing, feature engineering, preprocessing, and model.
"""

import pytest
from pipeline.model_integration import HealthConnectPipeline
from pipeline.validation import ValidationError


@pytest.fixture(scope="module")
def fitted_pipeline():
    """Fit the full pipeline once for all integration tests."""
    pipe = HealthConnectPipeline(threshold=0.50)
    pipe.run(model_name="Random Forest")
    return pipe


# ── Pipeline fit / predict tests ─────────────────────────────────────────────

def test_pipeline_fit_returns_results(fitted_pipeline):
    assert fitted_pipeline._fitted is True
    assert fitted_pipeline.model is not None
    assert fitted_pipeline.preprocessor is not None


def test_pipeline_feature_names_not_empty(fitted_pipeline):
    assert len(fitted_pipeline.feature_names) > 0


def test_pipeline_predict_single_returns_valid_output(fitted_pipeline):
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
    }
    result = fitted_pipeline.predict_single(record)
    assert "no_show_probability" in result
    assert "risk_tier" in result
    assert 0 <= result["no_show_probability"] <= 1
    assert result["risk_tier"] in {"HIGH_RISK", "MODERATE_RISK", "LOW_RISK"}


def test_pipeline_prediction_output_validates(fitted_pipeline):
    """Test that prediction DataFrame passes output validation."""
    import pandas as pd
    from pipeline.validation import validate_prediction_output
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    preds = pd.DataFrame({
        "appointment_id": [f"HC-{i:05d}" for i in range(10)],
        "no_show_probability": probs,
        "risk_tier": [fitted_pipeline._risk_tier(p) for p in probs],
    })
    validate_prediction_output(preds)


def test_pipeline_raises_if_not_fitted():
    pipe = HealthConnectPipeline()
    with pytest.raises(ValidationError, match="not been fitted"):
        pipe.predict_single({"gender": "Male"})
