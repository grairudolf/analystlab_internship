"""
HealthConnect Clinic — ML Pipeline Package
Week 5: Data Preprocessing, Feature Engineering & Baseline Model Training
"""

from pipeline.config import PIPELINE_CONFIG
from pipeline.data_processing import load_data, clean_data, build_preprocessing_pipeline
from pipeline.feature_engineering import engineer_features
from pipeline.model_training import get_baseline_models, train_models
from pipeline.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_feature_importance

__all__ = [
    "PIPELINE_CONFIG",
    "load_data",
    "clean_data",
    "build_preprocessing_pipeline",
    "engineer_features",
    "get_baseline_models",
    "train_models",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_feature_importance",
]
