# Week 6: HealthConnect Experience Lab (Machine Learning Engineering Track)

**Author:** Grai Rudolf
**Role:** Junior Machine Learning Engineer
**Track:** Machine Learning Engineering
**Project:** HealthConnect Clinic — Appointment No-Show Prediction
**Date:** September 2026

---

## Overview

Week 6 moves the HealthConnect ML Engineering track from initial implementation into
**integration, advanced development and validation**. Building on the Week 5 modular pipeline,
Week 6 delivers:

- **Improved model configurations** with tuned hyperparameters informed by Data Science error analysis.
- **Input and output validation checks** (`validation.py`) for schema and prediction correctness.
- **Proper logging** replacing print statements throughout the pipeline.
- **An integrated end-to-end pipeline** (`model_integration.py`) — single callable class wrapping load → clean → engineer → preprocess → train → predict.
- **Integration testing** validating the full pipeline flow.
- **Cross-track collaboration** with the Data Science track — integrating the candidate model and error analysis into the pipeline.

---

## Project Structure

```
week 6/
├── AnalystLab_Africa_Week6_Experience_Lab_Assignment.pdf
├── README.md
├── requirements.txt
├── week6_healthconnect_ml_pipeline_integration.ipynb
├── pipeline/
│   ├── __init__.py
│   ├── config.py              # Paths, logging, validation schema
│   ├── data_processing.py     # load, clean, preprocessing transformer
│   ├── feature_engineering.py
│   ├── model_training.py      # Baseline + improved model definitions
│   ├── model_integration.py   # End-to-end integrated pipeline class
│   ├── evaluation.py          # Metrics & visualisations
│   ├── validation.py          # Input/output validation checks
│   └── utils.py               # Seeds & helpers
├── figures/
│   ├── fig01_model_comparison_w6.png
│   ├── fig02_confusion_matrix_w6.png
│   ├── fig03_roc_curve_w6.png
│   ├── fig04_feature_importance_w6.png
│   └── fig05_w5_vs_w6_comparison.png
├── tests/
│   ├── test_pipeline.py       # Unit tests (12 tests)
│   └── test_integration.py    # Integration tests (5 tests)
└── reports/
```

---

## Key Week 6 Improvements

| Component | Week 5 | Week 6 |
|---|---|---|
| Validation | None | Schema + output validation |
| Logging | print() | Structured logging |
| Model integration | Separate train/eval | Single `HealthConnectPipeline` class |
| Testing | 6 unit tests | 12 unit + 5 integration tests |
| Cross-track | Identified dependency | Active integration with Data Science |

---

## How to Run

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
jupyter notebook week6_healthconnect_ml_pipeline_integration.ipynb
```

---

## Focus for Week 7

1. End-to-end validation with `GroupKFold` by `patient_id`.
2. Threshold tuning for $F_2$-score optimisation.
3. Model artifact loading and prediction on unseen data.
4. FastAPI inference service integration.
5. MLflow experiment tracking and Evidently AI drift monitoring.
