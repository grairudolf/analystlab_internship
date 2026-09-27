"""
Evaluation module for HealthConnect Week 8.
Computes comprehensive classification metrics, confusion matrices,
and demographic subgroup fairness audits (gender equality).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
from pipeline.config import logger


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Calculate core classification metrics on binary test predictions."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    logger.info(
        "[%s] Evaluation: ROC-AUC=%.4f | Recall=%.4f | Precision=%.4f | F1=%.4f | Brier=%.4f",
        model_name, metrics["roc_auc"], metrics["recall"], metrics["precision"], metrics["f1"], metrics["brier_score"]
    )
    return metrics


def audit_gender_fairness(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """
    Subgroup fairness analysis across gender categories (Female vs Male).
    Documents the known gender fairness gap flagged by Data Science (Mairame Samba Niang).
    """
    records = []
    genders = df_test["gender"].unique()

    for g in sorted(genders):
        mask = (df_test["gender"] == g).values
        if mask.sum() == 0:
            continue

        g_y_true = np.array(y_true)[mask]
        g_y_pred = np.array(y_pred)[mask]
        g_y_prob = np.array(y_prob)[mask]

        cm = confusion_matrix(g_y_true, g_y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            rec, prec, fpr = 0.0, 0.0, 0.0

        auc = roc_auc_score(g_y_true, g_y_prob) if len(np.unique(g_y_true)) > 1 else np.nan

        records.append({
            "gender": g,
            "sample_size": int(mask.sum()),
            "prevalence": round(float(g_y_true.mean()), 4),
            "recall": round(float(rec), 4),
            "precision": round(float(prec), 4),
            "fpr": round(float(fpr), 4),
            "roc_auc": round(float(auc), 4) if not np.isnan(auc) else None,
        })

    fairness_df = pd.DataFrame(records)
    logger.info("Fairness audit completed for gender subgroups:\n%s", fairness_df.to_string())
    return fairness_df
