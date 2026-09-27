"""
HealthConnect Machine Learning Pipeline Package — Week 8 Final Integration.
"""

from pipeline.config import (
    DATA_PATH,
    DEFAULT_THRESHOLD,
    RECOMMENDED_THRESHOLD_RANGE,
    RISK_TIERS,
    PREDICTION_INPUT_COLS,
    PIPELINE_CONFIG,
    logger,
)
from pipeline.data_processing import (
    load_data,
    clean_data,
    patient_level_split,
    build_preprocessing_pipeline,
    get_feature_names,
)
from pipeline.feature_engineering import engineer_features
from pipeline.model_training import (
    get_candidate_models,
    train_models_with_group_cv,
)
from pipeline.evaluation import evaluate_model, audit_gender_fairness
from pipeline.validation import (
    ValidationError,
    validate_raw_data,
    validate_clean_data,
    validate_features,
    validate_threshold,
    validate_prediction_input,
    validate_prediction_output,
)
from pipeline.model_integration import HealthConnectPipeline

__all__ = [
    "HealthConnectPipeline",
    "load_data",
    "clean_data",
    "patient_level_split",
    "build_preprocessing_pipeline",
    "get_feature_names",
    "engineer_features",
    "get_candidate_models",
    "train_models_with_group_cv",
    "evaluate_model",
    "audit_gender_fairness",
    "ValidationError",
    "validate_raw_data",
    "validate_clean_data",
    "validate_features",
    "validate_threshold",
    "validate_prediction_input",
    "validate_prediction_output",
    "DEFAULT_THRESHOLD",
    "RECOMMENDED_THRESHOLD_RANGE",
    "RISK_TIERS",
    "PREDICTION_INPUT_COLS",
    "PIPELINE_CONFIG",
    "logger",
]
