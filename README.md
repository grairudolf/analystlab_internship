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
└── week 4/                                                      # Week 4: HealthConnect Experience Lab Kickoff
    ├── README.md
    ├── AnalystLab_Africa_Week4_Experience_Lab_Assignment.pdf
    ├── HealthConnect_Appointment_Data.csv
    ├── HealthConnect_Clinic_Knowledge_Base.docx
    ├── HealthConnect_Data_Dictionary.xlsx
    ├── week4_healthconnect_ml_definition_system_design.ipynb   # Analysis Notebook
    ├── figures/                                                 # Diagnostic Figures & System Architecture
    └── reports/                                                 # LaTeX Source Files & Compiled PDFs
        ├── week4_ml_problem_definition_and_system_design.tex
        ├── week4_ml_problem_definition_and_system_design.pdf
        ├── week4_project_summary.tex
        └── week4_project_summary.pdf
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

---

## Week 3 Churn Model Results Summary

Evaluated on 20% Stratified Test Set ($N=1,405$):

| Model Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 80.43% | 66.44% | 52.69% | 0.5877 | 0.8400 | Linear baseline |
| **Decision Tree** | 79.43% | 61.96% | 57.80% | 0.5981 | 0.8317 | Rule generation |
| **Random Forest** | 79.64% | 65.25% | 49.46% | 0.5627 | 0.8359 | Standard bagging |
| **Gradient Boosting** | 79.93% | 65.63% | 50.81% | 0.5727 | 0.8364 | Standard boosting |
| **Support Vector Machine** | 79.72% | 69.16% | 42.20% | 0.5242 | 0.7836 | Margin optimization |
| **K-Nearest Neighbors** | 76.37% | 56.29% | 48.12% | 0.5188 | 0.7993 | Instance similarity |
| **Random Forest (Balanced)** | **77.01%** | **54.79%** | **75.27%** | **0.6342** | **0.8389** | **Recommended Model (High Recall)** |
| **Logistic Regression (Balanced)** | **74.38%** | **51.06%** | **77.42%** | **0.6154** | **0.8393** | High-Recall linear option |

---

## How to Run

```bash
# Clone repository
git clone <repo-url>
cd week_one

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm jupyter

# Open Week 4 Notebook
jupyter notebook "week 4/week4_healthconnect_ml_definition_system_design.ipynb"
```

---

## Deliverables Checklist

- [x] Week 1 EDA Notebook (`week1_churn_analysis.ipynb`)
- [x] Week 2 Preprocessing Notebook & CSV (`week2_data_preprocessing.ipynb`)
- [x] Week 3 Evaluation Notebook & PDF Reports (`week3_business_report.pdf`, `week3_model_evaluation_report.pdf`)
- [x] Week 4 HealthConnect Notebook (`week4_healthconnect_ml_definition_system_design.ipynb`)
- [x] 5 Diagnostic & System Architecture Visuals (`week 4/figures/`)
- [x] LaTeX PDF Reports (`reports/week4_ml_problem_definition_and_system_design.pdf`, `reports/week4_project_summary.pdf`)
- [x] Documentation (`README.md` & `week 4/README.md`)

---

## License

This repository is licensed under the MIT License.
