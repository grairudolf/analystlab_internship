# AnalystLab Africa — Machine Learning Internship Programme

**Intern:** Rudolf  
**Role:** Junior Machine Learning Engineer  
**Business Scenario:** ABC Communications Ltd — Customer Churn Prediction & Retention Optimization  

---

## 📌 Project Overview

This repository contains the complete end-to-end deliverables for the **AnalystLab Africa Machine Learning Internship Programme**. The project addresses customer churn for **ABC Communications Ltd**, transforming raw subscriber data into actionable retention strategies through rigorous Exploratory Data Analysis (EDA), Data Preprocessing & Feature Engineering, and Supervised Machine Learning Model Development & Performance Evaluation.

---

## 📂 Repository Structure

```
.
├── AnalystLab_Africa_Week3_Machine_Learning_Assignment(1).pdf  # Official Assignment Document
├── WA_Fn-UseC_-Telco-Customer-Churn.csv                         # Original Raw Dataset
├── week1_churn_analysis.ipynb                                   # Week 1: Business Understanding & EDA
├── week 2/
│   ├── README.md
│   ├── week2_data_preprocessing.ipynb                           # Week 2: Preprocessing & Feature Engineering
│   ├── telco_churn_processed.csv                                # Machine-Learning-Ready Dataset
│   ├── business_understanding_report.pdf
│   └── data_preprocessing_report.pdf
└── week 3/
    ├── README.md
    ├── week3_model_development_evaluation.ipynb                 # Week 3: Model Development & Evaluation
    ├── telco_churn_processed.csv                                # Model Input Dataset
    ├── week3_telco_churn_predictions.csv                        # Output Predictions & Probabilities
    ├── model_performance_summary.csv                            # Model Evaluation Metrics Summary
    ├── figures/                                                 # 11 Performance & Diagnostic Figures
    │   ├── fig01_confusion_matrices.png
    │   ├── fig02_roc_curves.png
    │   ├── fig03_precision_recall_curves.png
    │   ├── fig04_feature_importance_rf.png
    │   ├── fig05_feature_importance_gb.png
    │   ├── fig06_model_comparison_bar.png
    │   ├── fig07_f1_rocauc_comparison.png
    │   ├── fig08_churn_prediction_probability.png
    │   ├── fig09_learning_curves.png
    │   ├── fig10_threshold_optimization.png
    │   └── fig11_recall_precision_tradeoff.png
    └── reports/                                                 # LaTeX Source Files & PDF Deliverables
        ├── week3_business_report.tex
        ├── week3_business_report.pdf
        ├── week3_model_evaluation_report.tex
        └── week3_model_evaluation_report.pdf
```

---

## 📊 Summary of Weekly Deliverables

### Week 1: Business Understanding & Exploratory Data Analysis
- **Focus:** Problem framing, target variable inspection, demographic and service EDA.
- **Key Findings:** Overall churn rate is **26.54%** (1,869 / 7,043). Month-to-month contracts, fiber optic internet, electronic check payments, and short tenure exhibit the highest churn prevalence.
- **Deliverable:** `week1_churn_analysis.ipynb`.

### Week 2: Data Preprocessing & Feature Engineering
- **Focus:** Data type correction, missing value imputation, categorical encoding, scaling, and feature creation.
- **Engineered Features:** `AvgMonthlySpend`, `TenureGroup`, `NumStreamingServices`, `NumSecurityServices`, and `HasInternet`.
- **Deliverables:** `week 2/week2_data_preprocessing.ipynb`, `week 2/telco_churn_processed.csv`, `business_understanding_report.pdf`, `data_preprocessing_report.pdf`.

### Week 3: Machine Learning Model Development & Performance Evaluation
- **Focus:** Supervised classification model development, stratified train/test split, diagnostic visualizations, cost-sensitive optimization, LaTeX reports, and strategic recommendations.
- **Models Benchmarked:**
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. Gradient Boosting Classifier
  5. Support Vector Machine (SVM with probability calibration)
  6. K-Nearest Neighbors (KNN)
  7. Cost-Sensitive / Class-Balanced Variants (Random Forest & Logistic Regression)

---

## 🏆 Model Performance Benchmark Summary

Evaluated on an independent 20% Stratified Test Set ($N=1,405$):

| Model Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Primary Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 80.43% | 66.44% | 52.69% | 0.5877 | 0.8400 | Linear baseline |
| **Decision Tree** | 79.43% | 61.96% | 57.80% | 0.5981 | 0.8317 | Rule generation |
| **Random Forest** | 79.64% | 65.25% | 49.46% | 0.5627 | 0.8359 | Standard bagging |
| **Gradient Boosting** | 79.93% | 65.63% | 50.81% | 0.5727 | 0.8364 | Standard boosting |
| **Support Vector Machine** | 79.72% | 69.16% | 42.20% | 0.5242 | 0.7836 | Margin optimization |
| **K-Nearest Neighbors** | 76.37% | 56.29% | 48.12% | 0.5188 | 0.7993 | Instance similarity |
| 🌟 **Random Forest (Balanced)** | **77.01%** | **54.79%** | **75.27%** | **0.6342** | **0.8389** | **Recommended Production Deployment** |
| 🌟 **Logistic Regression (Balanced)** | **74.38%** | **51.06%** | **77.42%** | **0.6154** | **0.8393** | High-Recall linear option |

---

## 🎯 Key Business Insights & Deployment Recommendations

1. **Why Recall Over Accuracy?**
   - **Cost of False Negative (FN):** ~$600 (lost customer lifetime value).
   - **Cost of False Positive (FP):** ~$25 (cost of proactive retention offer).
   - Failing to detect a churning customer is **24x more costly** than sending a retention discount to a loyal customer. Class balancing and threshold adjustment boost Recall from **50.8% to 75.3%**, capturing 3 out of every 4 churning subscribers.

2. **Top Predictive Drivers of Churn:**
   - **Contract Type:** Month-to-month subscribers churn at 42.7% vs <3% for 2-year contract holders.
   - **Tenure:** Highest churn occurs within the first 12 months of service.
   - **Monthly Spend & Fiber Optic:** High monthly bills combined with fiber optic service friction trigger high churn risk.

3. **Financial Impact & Strategic ROI:**
   - Intercepting 75.3% of churners with targeted offers (assuming 35% retention acceptance) yields an estimated net annual financial gain exceeding **$340,000**.

---

## 🚀 How to Run

```bash
# Clone repository
git clone <repo-url>
cd week_one

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm reportlab pypdf jupyter

# Run Week 3 asset generator script
python generate_week3_assets.py

# Build LaTeX and PDF reports
python build_latex_and_pdf_reports.py

# Launch Jupyter Notebook
jupyter notebook week\ 3/week3_model_development_evaluation.ipynb
```

---

## 📄 Deliverables Checklist

- [x] Executed Jupyter Notebook (`week3_model_development_evaluation.ipynb`)
- [x] Processed Dataset & Predictions CSV (`week3_telco_churn_predictions.csv`)
- [x] 11 High-Resolution Diagnostic Figures (`figures/`)
- [x] Professional LaTeX Source Files (`week3_business_report.tex`, `week3_model_evaluation_report.tex`)
- [x] PDF Reports (`week3_business_report.pdf`, `week3_model_evaluation_report.pdf`)
- [x] Updated Project Documentation (`README.md`)
