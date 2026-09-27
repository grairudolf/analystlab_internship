# Week 8: HealthConnect Final Integration, Presentation & Project Showcase

**Author:** Grai Rudolf  
**Role:** Junior Machine Learning Engineer  
**Track:** Machine Learning Engineering  
**Project:** HealthConnect Clinic — Appointment No-Show Prediction & Clinical Action Engine  
**Date:** September 2026  
**Status:** Complete & Demonstration Ready (100% Tests Passing)  

---

## Overview

Week 8 represents the culmination of the **HealthConnect Experience Lab**. Throughout Weeks 4–7, the Machine Learning Engineering track advanced from problem framing, data leakage detection, and modular pipeline development to comprehensive testing and reliability hardening. 

In Week 8, we integrate all components into a production-grade, reproducible, and demonstrated engineering package that collaborates directly with our **HC-POD multidisciplinary team** (Data Science, Data Analytics, Generative AI, and Project Management).

### Core Week 8 Achievements
- **Zero Patient Contamination:** Enforced patient-level grouped splitting (`GroupShuffleSplit`) and 5-fold `GroupKFold` cross-validation on `patient_id` (1,680 unique patients) ensuring 100% unseen patient evaluation.
- **Leakage-Free Cross-Track Handoff:** Resolved post-arrival data leakage by barring `waiting_time_minutes` from training and adding schema guards at the API boundary.
- **Data Science Contract Compatibility:** Integrated candidate Random Forest model specifications from Data Science (Mairame Samba Niang) with dedicated adapter support (`import_ds_model()`).
- **Cost-Sensitive High Recall:** Operating threshold calibrated to **0.35**, achieving **94.20% Recall** on missed appointments to minimize doctor idle time and maximize clinic slot recovery.
- **Actionable Risk-Tier Routing:** Automated categorization into **HIGH_RISK** (WhatsApp + buffer slots), **MODERATE_RISK** (SMS reminders), and **LOW_RISK** (email).
- **Quality Assurance & Reliability:** 21 automated pytest tests (Unit, Integration, Reliability, HC-POD Contract) all passing in 17.8s.
- **Ethical AI & Fairness Audit:** Subgroup evaluation across gender groups transparently documenting sensitivity equality and a ~0.025 ROC-AUC disparity.

---

## Repository Structure

```
week 8/
├── README.md                                                   # This comprehensive documentation file
├── requirements.txt                                            # Environment dependencies
├── AnalystLab_Africa_Week8_Data_Analytics_Assignment(1).pdf   # Assignment specification
├── week8_healthconnect_final_integration.ipynb               # Fully executed showcase notebook
├── generate_figures.py                                         # Diagnostic figure generator script
├── generate_reports.py                                         # PDF report compiler script
├── build_and_run_notebook.py                                   # Automated notebook builder & runner
├── pipeline/                                                   # Core modular ML package
│   ├── __init__.py                                             # Package exports
│   ├── config.py                                               # Constants, paths, risk tiers, schemas
│   ├── data_processing.py                                      # Cleaning, imputation, grouped splitting
│   ├── feature_engineering.py                                  # Domain feature derivation
│   ├── model_training.py                                       # GroupKFold CV & benchmark training
│   ├── model_integration.py                                    # HealthConnectPipeline production class
│   ├── evaluation.py                                           # Metrics & gender fairness audit
│   ├── validation.py                                           # Strict schema validation & leak guards
│   └── utils.py                                                # Random seed and helper utilities
├── demo/                                                       # Interactive demonstration suite
│   └── demo_inference.py                                       # Live showcase demonstration script
├── tests/                                                      # 21 Automated pytest tests
│   ├── test_pipeline.py                                        # 8 Unit tests
│   ├── test_integration.py                                     # 4 Integration tests
│   ├── test_reliability.py                                     # 5 Reliability & error-handling tests
│   └── test_ds_contract.py                                     # 4 HC-POD Data Science contract tests
├── artifacts/                                                  # Serialized production pipeline
│   ├── model.pkl                                               # Fitted Random Forest classifier
│   ├── preprocessor.pkl                                        # Fitted ColumnTransformer
│   └── meta.json                                               # Pipeline configuration metadata
├── figures/                                                    # High-resolution diagnostic figures (300 DPI)
│   ├── fig01_ml_system_architecture_w8.png                     # End-to-end multidisciplinary architecture
│   ├── fig02_roc_curves_comparison_w8.png                      # ROC curves & threshold calibration
│   ├── fig03_confusion_matrix_w8.png                           # Confusion matrix at 0.35 threshold
│   ├── fig04_risk_tier_distribution_w8.png                     # Operational risk tier stratification
│   ├── fig05_test_suite_summary_w8.png                         # 21-test QA verification summary
│   └── fig06_gender_fairness_audit_w8.png                      # Subgroup fairness audit across genders
├── reports/                                                    # Compiled corporate PDF deliverables
│   ├── week8_final_ml_pipeline_integration_report.pdf         # Comprehensive 8-page technical report
│   └── week8_project_summary.pdf                               # Executive 2-page project summary
└── presentation/                                               # Showcase presentation materials
    ├── presentation_slides.md                                  # Complete 10-slide deck outline
    ├── presentation_script.md                                  # Word-for-word timed script (5–10 mins)
    └── slides.html                                             # Interactive browser slide viewer
```

---

## Model Evaluation on Unseen Patients

Evaluated on an independent patient-level test split ($N=966$ appointments, 336 patients):

| Metric | Logistic Regression (Baseline) | Random Forest (Final DS Spec) | Operational Impact |
| :--- | :---: | :---: | :--- |
| **ROC-AUC (Unseen Patients)** | 0.6735 | **0.6846** | +1.11% discriminative capability |
| **Operating Threshold** | 0.50 | **0.35** | Cost-sensitive high-recall calibration |
| **Recall on No-Shows** | 72.40% | **94.20%** | **Intercepts 19 out of every 20 missed visits** |
| **Precision** | 52.10% | **53.85%** | Consistent positive predictive value |
| **F1-Score** | 0.6058 | **0.6852** | Superior harmonic balance |
| **Brier Score** | 0.2412 | **0.2256** | Lower mean squared probability calibration error |

---

## Operational Risk Tiers & Clinical Interventions

Predictions map directly to clinical workflows via the `predict_single()` and `predict_batch()` APIs:

```json
{
  "appointment_id": "HC-00421",
  "no_show_probability": 0.7452,
  "risk_tier": "HIGH_RISK",
  "recommended_action": "TRIGGER_WHATSAPP_CONFIRMATION_AND_BUFFERSLOT"
}
```

1. **HIGH_RISK ($P \ge 0.70$):** Triggers multi-channel interactive WhatsApp confirmation bot and flags clinic scheduling software to allocate a contingent buffer slot.
2. **MODERATE_RISK ($0.35 \le P < 0.70$):** Dispatches automated 48-hour SMS reminder requesting a "1 to Confirm" reply.
3. **LOW_RISK ($P < 0.35$):** Routes to standard calendar email notification, conserving SMS telephony overhead.

---

## How to Run & Verify

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all 21 automated tests
PYTHONPATH="." pytest tests/ -v

# 3. Run interactive showcase demonstration
python demo/demo_inference.py

# 4. Generate all diagnostic figures & PDF reports
python generate_figures.py
python generate_reports.py

# 5. Open presentation slide deck in any browser
python3 -m http.server 8000
# Navigate to: http://localhost:8000/presentation/slides.html
```

---

## HC-POD Multidisciplinary Final Integration

| Integration Dimension | HC-POD Exchange Record (MLE $\leftrightarrow$ Data Science) |
| :--- | :--- |
| **Collaborator** | Data Science Track — Mairame Samba Niang |
| **Dependency** | Tuned candidate model algorithm, hyperparameters, and decision threshold. |
| **Output Provided by MLE** | 12-feature input contract, JSON output schema, pickle requirement, and leakage detection (`waiting_time_minutes`). |
| **Output Received from DS** | Retrained Random Forest model specs (ROC-AUC 0.682), threshold range (0.30–0.35), gender gap alert, and revised bootstrap CI finding. |
| **What Changed** | DS retrained model honestly without leakage; MLE updated pipeline threshold to 0.35, added GroupKFold, and built DS artifact import adapter. |
| **Improvement** | Eliminated deceptive leakage, increased recall to 94.2%, and ensured 100% runtime compatibility. |
| **Evidence** | Documented conversation logs, contract tests (`tests/test_ds_contract.py`), and commit history. |

---

## Known Limitations

1. **Gender Subgroup Fairness Disparity:** Auditing reveals an ROC-AUC gap: Female patients achieve 0.6966 AUC with 95.2% recall, while Male patients achieve 0.6719 AUC with 93.4% recall. Both groups have high recall (>93%), but clinic policy must ensure male patients are not penalized with disproportionate cancellation risks.
2. **Statistical Significance vs Baseline:** After removing data leakage from both models, the 95% bootstrap confidence interval of the Random Forest gain over Logistic Regression includes zero. The model delivers a small, favorable gain (+1.1% AUC); Random Forest is chosen for non-linear feature interaction capture rather than statistical dominance.
3. **Stationary Lead-Time Distribution:** The model assumes patient booking patterns remain stable; significant changes in clinic scheduling windows will necessitate retraining.
