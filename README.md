# AnalystLab Africa — Machine Learning Internship Repository

**Author:** Rudolf  
**Role:** Machine Learning Engineer Intern  
**Program:** AnalystLab Africa Internship  

---

## Overview

This repository contains my weekly projects, notebooks, code, and reports for the AnalystLab Africa Machine Learning Track.

- **Weeks 1–3:** Telecom Customer Churn Prediction for ABC Communications Ltd.
- **Week 4 onwards:** HealthConnect Experience Lab — Appointment No-Show Prediction & Production ML System Design.

---

## Repository Layout

```
.
├── README.md
├── WA_Fn-UseC_-Telco-Customer-Churn.csv                         # Raw Telco Churn Dataset
├── week 1/                                                      # Week 1: Churn EDA & Problem Understanding
│   ├── README.md
│   └── week1_churn_analysis.ipynb
├── week 2/                                                      # Week 2: Feature Engineering & Preprocessing
│   ├── README.md
│   ├── week2_data_preprocessing.ipynb
│   ├── telco_churn_processed.csv
│   ├── business_understanding_report.pdf
│   └── data_preprocessing_report.pdf
├── week 3/                                                      # Week 3: Model Development & Evaluation
│   ├── README.md
│   ├── week3_model_development_evaluation.ipynb
│   ├── telco_churn_processed.csv
│   ├── week3_telco_churn_predictions.csv
│   ├── model_performance_summary.csv
│   ├── figures/                                                 # Diagnostic & Performance Plots
│   └── reports/                                                 # Final PDF Reports
├── week 4/                                                      # Week 4: HealthConnect Experience Lab Kickoff
│   ├── README.md
│   ├── week4_healthconnect_ml_definition_system_design.ipynb
│   ├── figures/
│   └── reports/
├── week 5/                                                      # Week 5: Modular ML Pipeline Implementation
│   ├── README.md
│   ├── pipeline/
│   ├── tests/
│   └── week5_healthconnect_ml_pipeline.ipynb
├── week 6/                                                      # Week 6: Cross-Track Pipeline Integration & Validation
│   ├── README.md
│   ├── pipeline/
│   ├── tests/
│   ├── figures/
│   ├── reports/
│   └── week6_healthconnect_ml_pipeline_integration.ipynb
├── week 7/                                                      # Week 7: Pipeline Testing, Reliability & Refinement
│   ├── README.md
│   ├── pipeline/
│   ├── tests/
│   ├── figures/
│   ├── reports/
│   └── week7_healthconnect_ml_pipeline_testing.ipynb
└── week 8/                                                      # Week 8: Final Integration, Showcase Package & Presentation
    ├── README.md
    ├── requirements.txt
    ├── week8_healthconnect_final_integration.ipynb
    ├── pipeline/                                                # Modular ML pipeline (GroupKFold & risk tier engine)
    ├── demo/                                                    # Interactive live demonstration
    ├── tests/                                                   # 21 automated pytest tests
    ├── artifacts/                                               # Serialized model & preprocessor
    ├── figures/                                                 # 6 diagnostic figures (300 DPI)
    ├── reports/                                                 # 2 compiled PDF reports
    └── presentation/                                            # Slide deck & timed video presentation script
```

---

## Summary of Weekly Progress

### Week 1: Business Understanding & Churn EDA
- Conducted exploratory data analysis on 7,043 telecom subscriber records.
- Identified overall churn rate (26.54%) and key drivers: month-to-month contracts, fiber optic service, electronic check payments, and short tenure.
- **Notebook:** `week 1/week1_churn_analysis.ipynb`

### Week 2: Data Preprocessing & Feature Engineering
- Imputed missing values, scaled numeric features, and encoded categorical variables.
- Engineered domain features: `AvgMonthlySpend`, `TenureGroup`, `NumStreamingServices`, `NumSecurityServices`, and `HasInternet`.
- **Notebook & Reports:** `week 2/week2_data_preprocessing.ipynb`, processed dataset, and PDF reports.

### Week 3: Supervised Machine Learning & Cost-Sensitive Tuning
- Benchmarked 6 supervised classification algorithms on a 20% test split ($N=1,405$).
- Applied class balancing (cost-sensitive learning & threshold tuning) to prioritize Recall over Accuracy, increasing churner detection rate from 50.8% to 75.3%.
- **Notebook & Reports:** `week 3/week3_model_development_evaluation.ipynb`, predictions CSV, 11 figures, and PDF reports.

### Week 4: HealthConnect Experience Lab (ML Problem Framing & System Design)
- Transitioned to healthcare domain: analyzed 5,000 appointment records to frame the appointment no-show prediction problem ($48.46\%$ baseline no-show rate).
- Defined data leakage boundaries (excluding post-arrival `waiting_time_minutes`), missingness mechanisms (`reminder_channel` NMAR), and patient clustering rules (`GroupKFold` by `patient_id`).
- Designed a 5-stage production ML architecture (Ingestion -> Feature Store -> FastAPI Microservice -> Actionable Risk-Tier Decision Engine -> MLOps Monitoring).
- **Notebook & Reports:** `week 4/week4_healthconnect_ml_definition_system_design.ipynb`, 5 figures, LaTeX source files, and compiled PDFs.

### Week 5: Modular ML Pipeline Implementation
- Architected a modular Python ML pipeline (`pipeline/`: data processing, feature engineering, model training, evaluation, validation).
- Implemented baseline benchmarking, domain feature derivation, and initial test suite.
- **Notebook & Tests:** `week 5/week5_healthconnect_ml_pipeline.ipynb` and pytest unit tests.

### Week 6: Cross-Track Pipeline Integration & Validation
- Integrated pipeline components across tracks with structured logging and schema validation.
- Built end-to-end `HealthConnectPipeline` class with single-record and batch inference capabilities.
- **Notebook & Reports:** `week 6/week6_healthconnect_ml_pipeline_integration.ipynb`, 4 figures, and PDF reports.

### Week 7: Pipeline Testing, Reliability & Refinement
- Executed 31 automated tests (12 unit + 5 integration + 14 reliability tests).
- Added input validation for predictions, threshold validation, and artifact round-trip persistence.
- **Notebook, Tests & Reports:** `week 7/week7_healthconnect_ml_pipeline_testing.ipynb`, 31 tests, and PDF reports.

### Week 8: Final Integration, Presentation & Project Showcase (Culmination)
- Achieved **zero patient leakage** via patient-level grouped train/test splitting (`GroupShuffleSplit` on `patient_id`).
- Integrated Data Science handoff specifications (Random Forest, ROC-AUC = 0.6846 on unseen patients).
- Calibrated cost-sensitive operating threshold to **0.35**, achieving **94.20% Recall** on missed appointments.
- Built automated Risk-Tier Decision Engine (High, Moderate, Low Risk) mapped to clinical workflows (WhatsApp + buffer slot, SMS, email).
- Audited demographic fairness across gender groups, documenting sensitivity equality and ROC-AUC parity.
- 21 automated tests passing (100% pass rate in 17.8s), including HC-POD contract validation suite.
- **Deliverables:** `week 8/` containing executed notebook, 6 publication-ready figures, 2 compiled PDF reports, interactive showcase demo, test suite, and presentation package with timed video script.

---

## HealthConnect Week 8 Final Performance Summary

Evaluated on Unseen Patient Test Set ($N=966$ appointments, 336 patients):

| Model Algorithm | Accuracy | Precision | Recall (No-Show) | F1 Score | ROC-AUC | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (Baseline)** | 61.28% | 52.10% | 72.40% | 0.6058 | 0.6735 | Standard linear baseline |
| **Random Forest (Final DS Spec)** | **64.70%** | **53.85%** | **94.20%** | **0.6852** | **0.6846** | **Final Production Model (Threshold = 0.35)** |
| **Gradient Boosting** | 63.46% | 52.94% | 85.12% | 0.6528 | 0.6595 | Ensemble comparator |

---

## Deliverables Checklist

- [x] Week 1 EDA Notebook (`week1_churn_analysis.ipynb`)
- [x] Week 2 Preprocessing Notebook & CSV (`week2_data_preprocessing.ipynb`)
- [x] Week 3 Evaluation Notebook & PDF Reports (`week3_business_report.pdf`, `week3_model_evaluation_report.pdf`)
- [x] Week 4 HealthConnect Problem Definition & System Design (`week 4/`)
- [x] Week 5 HealthConnect ML Pipeline Implementation (`week 5/`)
- [x] Week 6 HealthConnect Pipeline Integration & Validation (`week 6/`)
- [x] Week 7 HealthConnect Pipeline Testing & Reliability Refinement (`week 7/`)
- [x] Week 8 HealthConnect Final Integration, Showcase Package & Presentation (`week 8/`)
  - [x] Final modular pipeline with patient-level grouping (`pipeline/`)
  - [x] Executed showcase Jupyter notebook (`week8_healthconnect_final_integration.ipynb`)
  - [x] 21 automated pytest tests (Unit, Integration, Reliability, Contract) (`tests/`)
  - [x] 6 Diagnostic figures at 300 DPI (`figures/`)
  - [x] 2 Compiled PDF reports (`reports/week8_final_ml_pipeline_integration_report.pdf`, `reports/week8_project_summary.pdf`)
  - [x] Interactive live showcase demo script (`demo/demo_inference.py`)
  - [x] Serialized model artifacts (`artifacts/`)
  - [x] Full presentation materials & timed video script (`presentation/`)
  - [x] Comprehensive documentation (`README.md` & `week 8/README.md`)

---

## License

This repository is licensed under the MIT License.

