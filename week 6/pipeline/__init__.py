"""
HealthConnect Clinic — ML Pipeline Package
Week 6: Integrated Pipeline with Validation, Logging, and Model Integration
"""

from pipeline.config import PIPELINE_CONFIG
from pipeline.data_processing import load_data, clean_data, build_preprocessing_pipeline
from pipeline.feature_engineering import engineer_features
from pipeline.model_training import get_baseline_models, get_improved_models, train_models
from pipeline.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from pipeline.validation import (
    ValidationError,
    validate_raw_data,
    validate_clean_data,
    validate_features,
    validate_prediction_output,
)
from pipeline.model_integration import HealthConnectPipeline

__all__ = [
    "PIPELINE_CONFIG",
    "load_data",
    "clean_data",
    "build_preprocessing_pipeline",
    "engineer_features",
    "get_baseline_models",
    "get_improved_models",
    "train_models",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_feature_importance",
    "ValidationError",
    "validate_raw_data",
    "validate_clean_data",
    "validate_features",
    "validate_prediction_output",
    "HealthConnectPipeline",
]
