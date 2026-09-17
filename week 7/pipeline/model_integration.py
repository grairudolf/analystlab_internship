"""
Integrated ML pipeline that wraps the full flow: raw data → prediction.
Week 7: Added error handling, input validation for predict methods,
threshold validation, and artifact loading support.
"""

import json
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from pipeline.config import (
    CATEGORICAL_COLS,
    DEFAULT_THRESHOLD,
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
    validate_feature_names,
    validate_features,
    validate_prediction_input,
    validate_prediction_output,
    validate_raw_data,
    validate_threshold,
)


class HealthConnectPipeline:
    """
    End-to-end integrated pipeline: load → clean → engineer → preprocess → train → predict.
    Week 7: Refined with input/output validation and error handling.
    """

    # Input columns required for a single prediction record (post-cleaning/feature-engineering)
    REQUIRED_INPUT_COLS = [
        "gender", "age", "age_group", "appointment_type", "appointment_day",
        "appointment_time", "booking_lead_days", "previous_appointments",
        "previous_no_shows", "reminder_sent", "reminder_channel",
        "distance_to_clinic_km",
    ]

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        validate_threshold(threshold)
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
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
            )
        except ValueError as e:
            logger.error("Train/test split failed: %s", e)
            raise ValidationError(f"Train/test split failed: {e}")
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
        try:
            models = get_improved_models()
            results = train_models(models, X_train_proc, y_train, cv=5)
        except Exception as e:
            logger.error("Model training failed: %s", e)
            raise ValidationError(f"Model training failed: {e}")

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
        validate_prediction_output(predictions)

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
        self._check_fitted()
        validate_prediction_input(record, required_cols=self.REQUIRED_INPUT_COLS)

        try:
            df = pd.DataFrame([record])
            df_proc = self.preprocessor.transform(df)
            prob = self.model.predict_proba(df_proc)[0, 1]
        except Exception as e:
            logger.error("Prediction failed for record: %s", e)
            raise ValidationError(f"Prediction failed: {e}")

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
        self._check_fitted()

        validate_prediction_input(df, required_cols=self.REQUIRED_INPUT_COLS)

        try:
            df_proc = self.preprocessor.transform(df)
            probs = self.model.predict_proba(df_proc)[:, 1]
        except Exception as e:
            logger.error("Batch prediction failed: %s", e)
            raise ValidationError(f"Batch prediction failed: {e}")

        preds = self._generate_predictions(df, df_proc, probs)
        validate_prediction_output(preds)
        return preds

    # ── Reproducibility check ────────────────────────────────────────────────
    def verify_reproducibility(self) -> bool:
        """
        Run a deterministic check that two fresh pipeline runs agree.
        Returns True if both runs select the same best model and feature count.
        """
        logger.info("Running reproducibility verification...")
        run1 = HealthConnectPipeline(threshold=self.threshold)
        r1 = run1.run()
        run2 = HealthConnectPipeline(threshold=self.threshold)
        r2 = run2.run()

        same_model = r1["best_model_name"] == r2["best_model_name"]
        same_features = len(run1.feature_names) == len(run2.feature_names)
        reproducible = same_model and same_features
        logger.info("Reproducibility: same_model=%s, same_features=%s → %s",
                    same_model, same_features, "PASS" if reproducible else "FAIL")
        return reproducible

    # ── Save / load model artifacts ──────────────────────────────────────────
    def save_artifacts(self, path: str = "artifacts") -> None:
        """Save the fitted pipeline (preprocessor + model) to disk."""
        if not self._fitted:
            raise ValidationError("Pipeline has not been fitted. Call run() first.")
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

    @classmethod
    def load_artifacts(cls, path: str = "artifacts") -> "HealthConnectPipeline":
        """Load a previously saved pipeline (preprocessor + model + meta)."""
        if not os.path.exists(os.path.join(path, "meta.json")):
            raise ValidationError(f"No pipeline artifacts found at {path}/")
        with open(os.path.join(path, "preprocessor.pkl"), "rb") as f:
            preprocessor = pickle.load(f)
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        pipe = cls(threshold=meta.get("threshold", DEFAULT_THRESHOLD))
        pipe.preprocessor = preprocessor
        pipe.model = model
        pipe.model_name = meta.get("model_name", "unknown")
        pipe.feature_names = meta.get("feature_names", [])
        pipe._fitted = True
        logger.info("Pipeline artifacts loaded from %s/ (model: %s)", path, pipe.model_name)
        return pipe

    # ── Private helpers ──────────────────────────────────────────────────────
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise ValidationError("Pipeline has not been fitted. Call run() or load_artifacts() first.")

    def _generate_predictions(self, X_raw, X_proc, probs) -> pd.DataFrame:
        """Create a DataFrame of predictions with risk tiers."""
        pred = pd.DataFrame({
            "appointment_id": X_raw["appointment_id"] if "appointment_id" in X_raw.columns else [f"HC-{i:05d}" for i in range(len(probs))],
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
