"""
Integrated ML pipeline that wraps the full flow: raw data → prediction.
Week 6: End-to-end pipeline with validation, logging, and configurable thresholds.
"""

import json
import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from pipeline.config import (
    CATEGORICAL_COLS,
    DEFAULT_THRESHOLD,
    ENGINEERED_CAT_COLS,
    ENGINEERED_NUM_COLS,
    NUMERICAL_COLS,
    RANDOM_STATE,
    TEST_SIZE,
    logger,
)
from pipeline.data_processing import (
    build_preprocessing_pipeline,
    clean_data,
    get_feature_names,
    load_data,
)
from pipeline.evaluation import evaluate_model
from pipeline.feature_engineering import engineer_features
from pipeline.model_training import get_improved_models, train_models
from pipeline.validation import (
    ValidationError,
    validate_clean_data,
    validate_features,
    validate_prediction_output,
    validate_raw_data,
    validate_feature_names,
)


class HealthConnectPipeline:
    """
    End-to-end integrated pipeline: load → clean → engineer → preprocess → train → predict.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.preprocessor: Optional[ColumnTransformer] = None
        self.model = None
        self.model_name: str = ""
        self.feature_names: list = []
        self._fitted = False

    # ── Full pipeline execution ──────────────────────────────────────────────
    def run(self, model_name: str = "Random Forest") -> dict:
        """Execute the full pipeline and return results dictionary."""
        logger.info("=" * 60)
        logger.info("HealthConnect Integrated Pipeline — Starting")
        logger.info("Model target: %s | Threshold: %.2f", model_name, self.threshold)
        logger.info("=" * 60)

        # 1. Load
        raw = load_data()
        validate_raw_data(raw)

        # 2. Clean
        cleaned = clean_data(raw)
        validate_clean_data(cleaned)

        # 3. Feature engineer
        featured = engineer_features(cleaned)
        expected = ["historical_noshow_ratio", "lead_time_bin", "is_new_patient", "has_reminder"]
        validate_features(featured, expected)

        # 4. Split
        X = featured.drop(columns=["target"])
        y = featured["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        logger.info(
            "Train/test split: train=%d, test=%d, no-show rate=%.3f",
            len(X_train), len(X_test), y.mean(),
        )

        # 5. Preprocess
        self.preprocessor = build_preprocessing_pipeline()
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)
        self.feature_names = get_feature_names(self.preprocessor)
        validate_feature_names(self.preprocessor)
        logger.info("Preprocessed feature matrix: %d features", X_train_proc.shape[1])

        # 6. Train
        models = get_improved_models()
        results = train_models(models, X_train_proc, y_train, cv=5)

        # 7. Select best model
        best_name = max(results, key=lambda n: results[n]["cv_mean"])
        self.model = results[best_name]["model"]
        self.model_name = best_name
        self._fitted = True
        logger.info("Best model selected: %s (CV Recall: %.4f)", best_name, results[best_name]["cv_mean"])

        # 8. Evaluate on test
        y_prob = self.model.predict_proba(X_test_proc)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        metrics = evaluate_model(y_test, y_pred, y_prob, best_name)

        # 9. Generate predictions with risk tiers
        predictions = self._generate_predictions(X_test, X_test_proc, y_prob)

        logger.info("=" * 60)
        logger.info("Pipeline execution complete.")
        logger.info("=" * 60)

        return {
            "results": results,
            "best_model_name": best_name,
            "test_metrics": metrics,
            "predictions": predictions,
            "X_test": X_test,
            "X_test_proc": X_test_proc,
            "y_test": y_test,
            "y_prob": y_prob,
        }

    # ── Single-record prediction ─────────────────────────────────────────────
    def predict_single(self, record: dict) -> dict:
        """Predict for a single appointment record (dict)."""
        if not self._fitted:
            raise ValidationError("Pipeline has not been fitted. Call run() first.")

        df = pd.DataFrame([record])
        df_proc = self.preprocessor.transform(df)
        prob = self.model.predict_proba(df_proc)[0, 1]
        risk = self._risk_tier(prob)

        return {
            "appointment_id": record.get("appointment_id", "UNKNOWN"),
            "no_show_probability": round(float(prob), 4),
            "risk_tier": risk,
            "recommended_action": self._recommended_action(risk),
        }

    # ── Batch predictions ────────────────────────────────────────────────────
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate risk-stratified predictions for a batch of records."""
        if not self._fitted:
            raise ValidationError("Pipeline has not been fitted. Call run() first.")

        df_proc = self.preprocessor.transform(df)
        probs = self.model.predict_proba(df_proc)[:, 1]
        return self._generate_predictions(df, df_proc, probs)

    # ── Save / load model artifacts ──────────────────────────────────────────
    def save_artifacts(self, path: str = "artifacts") -> None:
        """Save the fitted pipeline (preprocessor + model) to disk."""
        import os
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "preprocessor.pkl"), "wb") as f:
            pickle.dump(self.preprocessor, f)
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        meta = {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Pipeline artifacts saved to %s/", path)

    # ── Private helpers ──────────────────────────────────────────────────────
    def _generate_predictions(self, X_raw, X_proc, probs) -> pd.DataFrame:
        """Create a DataFrame of predictions with risk tiers."""
        pred = pd.DataFrame({
            "appointment_id": X_raw["appointment_id"] if "appointment_id" in X_raw.columns else range(len(probs)),
            "no_show_probability": probs,
            "risk_tier": [self._risk_tier(p) for p in probs],
            "recommended_action": [self._recommended_action(self._risk_tier(p)) for p in probs],
        })
        return pred

    @staticmethod
    def _risk_tier(prob: float) -> str:
        if prob >= 0.70:
            return "HIGH_RISK"
        elif prob >= 0.40:
            return "MODERATE_RISK"
        return "LOW_RISK"

    @staticmethod
    def _recommended_action(tier: str) -> str:
        actions = {
            "HIGH_RISK": "TRIGGER_WHATSAPP_CONFIRMATION_AND_BUFFERSLOT",
            "MODERATE_RISK": "TRIGGER_SMS_REMINDER_REQUESTING_CONFIRMATION",
            "LOW_RISK": "STANDARD_EMAIL_NOTIFICATION",
        }
        return actions.get(tier, "STANDARD_EMAIL_NOTIFICATION")
