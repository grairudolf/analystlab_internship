"""
Baseline and improved model definitions and training utilities.
Week 6: Added improved models (tuned Random Forest, Gradient Boosting with GridSearch).
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from pipeline.config import RANDOM_STATE, CV_FOLDS, logger

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def get_baseline_models() -> dict:
    """Return a dictionary of Week 5 baseline classifiers."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    return models


def get_improved_models() -> dict:
    """
    Return improved model configurations for Week 6.
    Tuned based on Week 5 error analysis: higher n_estimators, adjusted depth,
    and cost-sensitive learning to favour recall.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.5,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            random_state=RANDOM_STATE,
        ),
    }
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            min_child_samples=20,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    return models


def train_models(models: dict, X_train, y_train, cv: int = CV_FOLDS) -> dict:
    """
    Cross-validate each model and return fitted results.
    Returns dict of {name: {"model": fitted_estimator, "cv_scores": array, "cv_mean": float}}.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in models.items():
        logger.info("Training %s...", name)
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="recall")
        model.fit(X_train, y_train)
        results[name] = {
            "model": model,
            "cv_scores": cv_scores,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }
        logger.info("  CV Recall: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std())

    return results
