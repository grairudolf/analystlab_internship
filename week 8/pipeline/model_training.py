"""
Model training and benchmarking for HealthConnect Week 8.
Supports patient-grouped cross-validation (GroupKFold) and trains models matching
Data Science handoff specifications (Random Forest, tuned hyperparameters).
"""

from typing import Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

from pipeline.config import RANDOM_STATE, CV_FOLDS, logger


def get_candidate_models() -> Dict[str, Any]:
    """
    Define benchmark models:
    - Logistic Regression: Baseline model
    - Random Forest: Primary candidate model matching Data Science Week 7/8 specification
    - Gradient Boosting: Ensemble comparator
    """
    return {
        "Logistic Regression (Baseline)": LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest (Final DS Spec)": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
    }


def train_models_with_group_cv(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_splits: int = CV_FOLDS,
    scoring: str = "roc_auc",
) -> Dict[str, Dict[str, Any]]:
    """
    Perform cross-validation using GroupKFold grouped by patient_id,
    then fit each model on the full training set.
    """
    results = {}
    gkf = GroupKFold(n_splits=n_splits)

    for name, model in models.items():
        logger.info("Evaluating %s with %d-fold GroupKFold CV (scoring=%s)...", name, n_splits, scoring)
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=gkf,
            groups=groups,
            scoring=scoring,
            n_jobs=-1,
        )

        # Fit on entire training matrix
        model.fit(X_train, y_train)

        results[name] = {
            "model": model,
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
        }
        logger.info(
            "[%s] Group CV %s: %.4f (+/- %.4f)",
            name, scoring, results[name]["cv_mean"], results[name]["cv_std"]
        )

    return results
