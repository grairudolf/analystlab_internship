"""
Model evaluation metrics and visualization utilities.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pipeline.config import FIGURES_DIR


def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model") -> dict:
    """Compute and print a full classification evaluation report."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f2": fbeta_score(y_true, y_pred, beta=2),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Attended", "No-Show"]))
    for k, v in metrics.items():
        print(f"  {k:>12s}: {v:.4f}")

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """Plot and optionally save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Attended", "No-Show"], cmap="Blues", ax=ax
    )
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_roc_curve(y_true, y_prob, model_name="Model", save_path=None):
    """Plot and optionally save an ROC curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=model_name)
    ax.set_title(f"{model_name} — ROC Curve")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_feature_importance(feature_names, importances, model_name="Model", top_n=15, save_path=None):
    """Plot top N feature importances as a horizontal bar chart."""
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(indices)), importances[indices], color="#2E86C1")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_model_comparison(results: dict, save_path=None):
    """Bar chart comparing recall across all baseline models."""
    names = list(results.keys())
    recalls = [results[n]["cv_mean"] for n in names]
    stds = [results[n]["cv_std"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, recalls, yerr=stds, color="#2E86C1", capsize=5)
    ax.set_ylabel("CV Recall")
    ax.set_title("Baseline Model Recall Comparison (5-Fold Stratified CV)")
    ax.set_ylim(0, 1.0)
    for bar, r in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{r:.3f}", ha="center")
    plt.xticks(rotation=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
