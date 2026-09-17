# Week 7: HealthConnect Experience Lab (Machine Learning Engineering Track)

**Author:** Grai Rudolf
**Role:** Junior Machine Learning Engineer
**Track:** Machine Learning Engineering
**Project:** HealthConnect Clinic — Appointment No-Show Prediction
**Date:** September 2026

---

## Overview

Week 7 moves the HealthConnect ML Engineering track into **systematic testing, reliability
refinement, and validation** of the integrated pipeline built in Week 6.

Week 7 delivers:
- **Comprehensive test suite** — 31 tests (12 unit + 5 integration + 14 reliability), all passing.
- **Error-handling improvements** — input validation for predictions, threshold validation,
  and artifact loading.
- **Evidence-based refinement cycle** — issues found via testing, fixed, and re-tested.
- **Reproducibility verification** — two independent pipeline runs agree.
- **Cross-track testing with Data Science** — pipeline handling of model inputs/outputs validated.

---

## Project Structure

```
week 7/
├── AnalystLab_Africa_Week7_Data_Analytics_Assignment(1).pdf   # Assignment (not committed)
├── README.md                                                   # This file
├── requirements.txt
├── week7_healthconnect_ml_pipeline_testing.ipynb              # Executed testing notebook
├── pipeline/
│   ├── __init__.py
│   ├── config.py              # Paths, logging, validation schema
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_integration.py   # + load_artifacts, input validation, error handling
│   ├── evaluation.py
│   ├── validation.py          # + validate_prediction_input, validate_threshold
│   └── utils.py
├── figures/
│   ├── fig01_confusion_matrix_w7.png
│   ├── fig02_roc_curve_w7.png
│   └── fig03_test_suite_summary.png
├── tests/
│   ├── test_pipeline.py       # 12 unit tests
│   ├── test_integration.py    # 5 integration tests
│   └── test_reliability.py    # 14 reliability tests (NEW in Week 7)
└── reports/
    ├── week7_ml_pipeline_testing_refinement.pdf
    └── week7_project_summary.pdf
```

---

## Week 7 Refinements

| # | Refinement | Purpose |
|---|---|---|
| R1 | `validate_prediction_input()` | Reject missing-column / null prediction inputs with clear errors |
| R2 | `validate_threshold()` | Reject out-of-range or non-numeric decision thresholds |
| R3 | `load_artifacts()` | Inverse of `save_artifacts()` for pipeline persistence |
| R4 | Error handling around split/train/predict | Informative `ValidationError` instead of cryptic stack traces |
| R5 | `verify_reproducibility()` | Confirm two runs with fixed seeds agree |

---

## Running the Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v        # all 31 tests
jupyter notebook week7_healthconnect_ml_pipeline_testing.ipynb
```

---

## Next Steps (Week 8)

1. Deploy the validated pipeline behind a FastAPI inference service (containerised).
2. Implement `GroupKFold` by `patient_id` as final leakage control.
3. Add MLflow experiment tracking and Evidently AI drift monitoring.
4. Final end-to-end validation across all HealthConnect components.
