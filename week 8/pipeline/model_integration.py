"""
Integrated production-ready ML pipeline for HealthConnect Week 8.
Wraps the entire workflow: raw data ingestion → leakage elimination → patient-level
grouped split → feature engineering → robust preprocessing → model training & group CV →
risk stratification & action routing → artifact persistence.
Supports direct handoff integration with Data Science (Mairame Samba Niang).
"""

import json
import os
import pickle
from typing import Optional, Dict, Any, List, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from pipeline.config import (
    DEFAULT_THRESHOLD,
    PREDICTION_INPUT_COLS,
    RISK_TIERS,
    GROUP_COL,
    ARTIFACTS_DIR,
    logger,
)
from pipeline.data_processing import (
    build_preprocessing_pipeline,
    clean_data,
    get_feature_names,
    load_data,
    patient_level_split,
)
from pipeline.evaluation import evaluate_model, audit_gender_fairness
from pipeline.feature_engineering import engineer_features
from pipeline.model_training import get_candidate_models, train_models_with_group_cv
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
    Production-grade ML pipeline for appointment no-show prediction and clinical action dispatch.
    Week 8: Incorporates patient-level GroupKFold split, cost-sensitive thresholding (0.35),
    Data Science artifact compatibility adapter, and demographic fairness auditing.
    """

    REQUIRED_INPUT_COLS = PREDICTION_INPUT_COLS

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        validate_threshold(threshold)
        self.threshold = threshold
        self.preprocessor: Optional[ColumnTransformer] = None
        self.model = None
        self.model_name: str = ""
        self.feature_names: List[str] = []
        self.evaluation_metrics: Dict[str, Any] = {}
        self.fairness_metrics: Optional[pd.DataFrame] = None
        self._fitted = False
        self._is_external_ds_pipeline = False

    # ── Full pipeline training & validation flow ─────────────────────────────
    def run(self, model_name: str = "Random Forest (Final DS Spec)") -> Dict[str, Any]:
        """Execute the end-to-end training and evaluation cycle."""
        logger.info("=" * 65)
        logger.info("HealthConnect Integrated Pipeline [Week 8 Final] — Initiating Run")
        logger.info("Selected Target Model: %s | Decision Threshold: %.2f", model_name, self.threshold)
        logger.info("=" * 65)

        # 1. Ingestion
        raw = load_data()
        validate_raw_data(raw)

        # 2. Cleaning & Leakage Prevention
        cleaned = clean_data(raw)
        validate_clean_data(cleaned)

        # 3. Domain Feature Engineering
        featured = engineer_features(cleaned)
        expected = ["historical_noshow_ratio", "lead_time_bin", "is_new_patient", "has_reminder"]
        validate_features(featured, expected)

        # 4. Patient-Level Grouped Split (Zero Patient Contamination)
        train_df, test_df = patient_level_split(featured, group_col=GROUP_COL)

        X_train = train_df.drop(columns=["target", "patient_id", "appointment_id"])
        y_train = train_df["target"].values
        groups_train = train_df[GROUP_COL].values

        X_test = test_df.drop(columns=["target", "patient_id", "appointment_id"])
        y_test = test_df["target"].values

        # 5. Preprocessing Pipeline Construction
        self.preprocessor = build_preprocessing_pipeline()
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)
        self.feature_names = get_feature_names(self.preprocessor)
        validate_feature_names(self.preprocessor)

        # 6. GroupKFold Model Benchmarking
        candidate_models = get_candidate_models()
        training_results = train_models_with_group_cv(
            candidate_models,
            X_train_proc,
            y_train,
            groups=groups_train,
            n_splits=5,
            scoring="roc_auc",
        )

        if model_name not in training_results:
            raise ValidationError(f"Requested model '{model_name}' not found. Available: {list(training_results.keys())}")

        self.model = training_results[model_name]["model"]
        self.model_name = model_name
        self._fitted = True

        # 7. Out-of-Sample Test Evaluation
        y_prob = self.model.predict_proba(X_test_proc)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        self.evaluation_metrics = evaluate_model(y_test, y_pred, y_prob, model_name=model_name)

        # 8. Demographic Fairness Audit (Gender Gap Analysis)
        self.fairness_metrics = audit_gender_fairness(test_df, y_test, y_pred, y_prob)

        # 9. Risk Stratification & Clinical Action Routing
        predictions = self._generate_predictions(test_df, y_prob)
        validate_prediction_output(predictions)

        logger.info("=" * 65)
        logger.info("Pipeline execution completed successfully.")
        logger.info("=" * 65)

        return {
            "training_results": training_results,
            "best_model_name": model_name,
            "test_metrics": self.evaluation_metrics,
            "fairness_metrics": self.fairness_metrics,
            "predictions": predictions,
            "train_df": train_df,
            "test_df": test_df,
            "X_test": X_test,
            "X_test_proc": X_test_proc,
            "y_test": y_test,
            "y_prob": y_prob,
        }

    # ── Single-Record Prediction (API & Microservice Interface) ──────────────
    def predict_single(self, record: dict) -> dict:
        """
        Produce a real-time risk prediction for a single appointment record.
        Strictly conforms to the JSON contract established with Data Science & clinic systems:
        {
            "appointment_id": str,
            "no_show_probability": float,
            "risk_tier": str,
            "recommended_action": str
        }
        """
        self._check_fitted()
        validate_prediction_input(record, required_cols=self.REQUIRED_INPUT_COLS)

        df = pd.DataFrame([record])
        df_feat = engineer_features(df)

        try:
            if self._is_external_ds_pipeline:
                # If an external complete pipeline was provided by Data Science
                prob = float(self.model.predict_proba(df_feat)[:, 1][0])
            else:
                proc = self.preprocessor.transform(df_feat)
                prob = float(self.model.predict_proba(proc)[:, 1][0])
        except Exception as e:
            logger.error("Single prediction failed: %s", e)
            raise ValidationError(f"Prediction execution error: {e}")

        risk_tier = self._risk_tier(prob)
        action = self._recommended_action(risk_tier)

        return {
            "appointment_id": record.get("appointment_id", "HC-UNKNOWN"),
            "no_show_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "recommended_action": action,
        }

    # ── Batch Prediction ─────────────────────────────────────────────────────
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process batch of appointment records and return risk-stratified decisions."""
        self._check_fitted()
        validate_prediction_input(df, required_cols=self.REQUIRED_INPUT_COLS)

        df_feat = engineer_features(df)
        try:
            if self._is_external_ds_pipeline:
                probs = self.model.predict_proba(df_feat)[:, 1]
            else:
                proc = self.preprocessor.transform(df_feat)
                probs = self.model.predict_proba(proc)[:, 1]
        except Exception as e:
            logger.error("Batch prediction failed: %s", e)
            raise ValidationError(f"Batch prediction execution error: {e}")

        predictions = self._generate_predictions(df, probs)
        validate_prediction_output(predictions)
        return predictions

    # ── Data Science Artifact Adapter ────────────────────────────────────────
    def import_ds_model(self, model_or_pipeline, is_full_pipeline: bool = False, model_name: str = "DS_Random_Forest") -> None:
        """
        Adapter method allowing seamless drop-in of Mairame's Data Science handoff artifact.
        If is_full_pipeline=True, the artifact wraps preprocessing + classifier.
        """
        self.model = model_or_pipeline
        self.model_name = model_name
        self._is_external_ds_pipeline = is_full_pipeline
        self._fitted = True
        logger.info("Successfully imported external Data Science model artifact: %s (full_pipeline=%s)", model_name, is_full_pipeline)

    # ── Artifact Persistence & Round-Trip ────────────────────────────────────
    def save_artifacts(self, path: str = ARTIFACTS_DIR) -> None:
        """Serialize pipeline components and metadata for deployment."""
        if not self._fitted:
            raise ValidationError("Cannot save un-fitted pipeline.")

        os.makedirs(path, exist_ok=True)

        if self.preprocessor is not None:
            with open(os.path.join(path, "preprocessor.pkl"), "wb") as f:
                pickle.dump(self.preprocessor, f)

        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        meta = {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "is_external_ds_pipeline": self._is_external_ds_pipeline,
            "evaluation_metrics": self.evaluation_metrics,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Pipeline artifacts successfully exported to: %s", path)

    @classmethod
    def load_artifacts(cls, path: str = ARTIFACTS_DIR) -> "HealthConnectPipeline":
        """Deserialize stored pipeline components into a fully functional instance."""
        meta_path = os.path.join(path, "meta.json")
        model_path = os.path.join(path, "model.pkl")
        preproc_path = os.path.join(path, "preprocessor.pkl")

        if not os.path.exists(meta_path) or not os.path.exists(model_path):
            raise ValidationError(f"Missing required artifact files in: {path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        preprocessor = None
        if os.path.exists(preproc_path):
            with open(preproc_path, "rb") as f:
                preprocessor = pickle.load(f)

        pipe = cls(threshold=meta.get("threshold", DEFAULT_THRESHOLD))
        pipe.model = model
        pipe.preprocessor = preprocessor
        pipe.model_name = meta.get("model_name", "LoadedModel")
        pipe.feature_names = meta.get("feature_names", [])
        pipe._is_external_ds_pipeline = meta.get("is_external_ds_pipeline", False)
        pipe.evaluation_metrics = meta.get("evaluation_metrics", {})
        pipe._fitted = True

        logger.info("Loaded pipeline artifact successfully from: %s (Model: %s)", path, pipe.model_name)
        return pipe

    # ── Reproducibility Verification ─────────────────────────────────────────
    def verify_reproducibility(self) -> bool:
        """
        Verify that independent runs with fixed random state generate identical
        test metrics within strict numerical tolerances.
        """
        logger.info("Commencing reproducibility verification across two independent runs...")
        p1 = HealthConnectPipeline(threshold=self.threshold)
        res1 = p1.run(model_name="Random Forest (Final DS Spec)")

        p2 = HealthConnectPipeline(threshold=self.threshold)
        res2 = p2.run(model_name="Random Forest (Final DS Spec)")

        auc1 = res1["test_metrics"]["roc_auc"]
        auc2 = res2["test_metrics"]["roc_auc"]
        match = abs(auc1 - auc2) < 1e-4

        logger.info("Reproducibility check: Run 1 AUC=%.4f, Run 2 AUC=%.4f -> %s", auc1, auc2, "PASSED" if match else "FAILED")
        return match

    # ── Internal Helpers ─────────────────────────────────────────────────────
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise ValidationError("Pipeline is not fitted. Please run(). or load_artifacts().")

    def _generate_predictions(self, df_raw: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
        """Generate structured prediction DataFrame with clinical risk tiers and recommended actions."""
        if "appointment_id" in df_raw.columns:
            appt_ids = df_raw["appointment_id"].values
        else:
            appt_ids = [f"HC-{i:05d}" for i in range(len(probs))]

        tiers = [self._risk_tier(p) for p in probs]
        actions = [self._recommended_action(t) for t in tiers]

        return pd.DataFrame({
            "appointment_id": appt_ids,
            "no_show_probability": np.round(probs, 4),
            "risk_tier": tiers,
            "recommended_action": actions,
        })

    @staticmethod
    def _risk_tier(prob: float) -> str:
        if prob >= RISK_TIERS["HIGH"]["threshold"]:
            return RISK_TIERS["HIGH"]["label"]
        elif prob >= RISK_TIERS["MODERATE"]["threshold"]:
            return RISK_TIERS["MODERATE"]["label"]
        return RISK_TIERS["LOW"]["label"]

    @staticmethod
    def _recommended_action(tier: str) -> str:
        for t_info in RISK_TIERS.values():
            if t_info["label"] == tier:
                return t_info["action"]
        return "STANDARD_EMAIL_NOTIFICATION"
