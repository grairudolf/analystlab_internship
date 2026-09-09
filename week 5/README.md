# Week 5: HealthConnect Experience Lab (Machine Learning Engineering Track)

**Author:** Grai Rudolf
**Role:** Junior Machine Learning Engineer
**Track:** Machine Learning Engineering
**Project:** HealthConnect Clinic — Appointment No-Show Prediction
**Date:** September 2026

---

## Overview

Week 5 moves the HealthConnect Experience Lab from planning into practical implementation. Building
directly on the Week 4 problem definition and 5-stage production system design, I delivered a
modular, reusable Machine Learning pipeline covering:

- Data loading, cleaning and explicit data-leakage control.
- A reusable scikit-learn preprocessing transformer (imputation + scaling + encoding).
- Behavioural feature engineering.
- Baseline model development and benchmarking (5 classifiers, 5-fold stratified CV).
- Initial evaluation with classification metrics and diagnostic visualisations.
- Component testing and pipeline documentation.

The work deliberately focuses on establishing a **reliable baseline and initial implementation**, not
final deployment (which is deferred to Week 6).

---

## Project Structure

```
week5/
├── AnalystLab_Africa_Week5_Experience_Lab_Assignment.pdf   # Assignment & guidelines
├── README.md                                                # This file
├── requirements.txt                                         # Pinned dependencies
├── week5_healthconnect_ml_pipeline.ipynb                    # Main executed notebook
├── pipeline/                                                # Modular ML pipeline package
│   ├── __init__.py
│   ├── config.py            # Paths, column lists, hyperparameters
│   ├── data_processing.py   # load_data, clean_data, preprocessing transformer
│   ├── feature_engineering.py
│   ├── model_training.py    # Baseline model definitions & CV training
│   ├── evaluation.py        # Metrics & visualisation utilities
│   └── utils.py             # Seeds & helpers
├── figures/                                                 # Generated diagnostic figures
│   ├── fig01_model_comparison.png
│   ├── fig02_confusion_matrix.png
│   ├── fig03_roc_curve.png
│   └── fig04_feature_importance.png
├── tests/
│   └── test_pipeline.py     # Component smoke tests (6 tests)
└── reports/                                                 # LaTeX reports & compiled PDFs
    ├── week5_ml_pipeline_implementation.tex
    ├── week5_ml_pipeline_implementation.pdf
    ├── week5_project_summary.tex
    └── week5_project_summary.pdf
```

---

## Key Technical Decisions

1. **Pipeline modularity:** The pipeline is decomposed into single-responsibility modules
   (`config`, `data_processing`, `feature_engineering`, `model_training`, `evaluation`) so each
   stage can be reused, tested, and later containerised — consistent with the Week 4 microservice
   architecture.

2. **Leakage control:** `waiting_time_minutes`, `patient_id`, and identifiers are dropped; only
   pre-appointment features feed the model. `Cancelled` records are excluded (binary Attended vs
   No-Show task).

3. **Imputation:**
   - `reminder_channel` (NMAR) → explicit `"None_Sent"` category.
   - `distance_to_clinic_km` (MCAR) → median.

4. **Feature engineering:** Added `historical_noshow_ratio`, `lead_time_bin`,
   `is_new_patient`, and `has_reminder`.

5. **Train/test strategy:** Stratified 80/20 split for the initial baseline. A
   `GroupKFold` (by `patient_id`) is flagged as the rigorous Week 6 refinement to fully control
   repeat-patient leakage.

6. **Primary baseline:** Logistic Regression (interpretable) — held-out recall **0.650**,
   accuracy **0.617**, ROC-AUC **0.664**.

---

## Baseline Results (held-out test set, N=948)

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|----------|-----------|--------|-------|---------|
| Logistic Regression | 0.617    | 0.620     | 0.650  | 0.634 | 0.664   |
| Random Forest       | 0.617    | 0.623     | 0.639  | 0.631 | 0.652   |
| Decision Tree       | 0.571    | 0.587     | 0.540  | 0.563 | 0.619   |
| Gradient Boosting   | 0.588    | 0.593     | 0.621  | 0.606 | 0.613   |
| LightGBM            | 0.593    | 0.601     | 0.606  | 0.604 | 0.622   |

---

## Testing

Run the pipeline component tests:

```bash
python -m pytest tests/ -v
```

All 6 tests pass.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Open/run the notebook
jupyter notebook week5_healthconnect_ml_pipeline.ipynb

# Run tests
python -m pytest tests/ -v
```

---

## Focus for Week 6

1. Rigorous leakage-controlled validation (`GroupKFold` by `patient_id`, temporal split).
2. Hyperparameter and decision-threshold tuning for the recall-driven objective ($F_2$ score).
3. Wrap the best model in the containerised FastAPI inference service from the Week 4 architecture.
4. Add MLflow experiment tracking and Evidently AI drift monitoring.
