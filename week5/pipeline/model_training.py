"""
Baseline model definitions and training utilities.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from pipeline.config import RANDOM_STATE

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def get_baseline_models() -> dict:
    """Return a dictionary of baseline classifiers."""
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


def train_models(models: dict, X_train, y_train, cv: int = 5) -> dict:
    """
    Cross-validate each model and return fitted results.
    Returns dict of {name: {"model": fitted_estimator, "cv_scores": array, "cv_mean": float}}.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="recall")
        model.fit(X_train, y_train)
        results[name] = {
            "model": model,
            "cv_scores": cv_scores,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }
        print(f"  CV Recall: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return results
