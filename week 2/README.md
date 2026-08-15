# AnalystLab Africa - Machine Learning Internship Programme

## Week 2: Data Preprocessing and Feature Engineering for Machine Learning

**Intern:** Rudolf
**Business Scenario:** ABC Communications Ltd - Customer Churn Prediction

## Project Overview

This repository contains the Week 2 deliverable for the AnalystLab Africa Machine Learning
Internship Programme. Building on the Week 1 business understanding and exploratory data
analysis, this project transforms the raw Telco Customer Churn dataset into a clean,
fully numeric, machine-learning-ready dataset through a documented preprocessing pipeline.

## Dataset

`WA_Fn-UseC_-Telco-Customer-Churn.csv` - the same dataset used in Week 1, containing 7,043
customer records across 21 columns, with `Churn` as the binary target variable.

## Repository Structure

```
.
├── README.md
├── week2_data_preprocessing.ipynb        # Full preprocessing pipeline notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original raw dataset
├── telco_churn_processed.csv             # Final machine-learning-ready dataset
├── reports/
│   ├── business_understanding_report.pdf
│   └── data_preprocessing_report.pdf
└── figures/                              # Exported visualizations used in the reports
```

## Preprocessing Pipeline Summary

1. **Data Inspection** - shape, dtypes, missing values, duplicates, descriptive statistics.
2. **Data Cleaning** - corrected `TotalCharges` data type, imputed 11 hidden missing values,
   standardized redundant category labels, removed the `customerID` identifier.
3. **Feature Engineering** - created `AvgMonthlySpend`, `TenureGroup`, `NumStreamingServices`,
   `NumSecurityServices`, and `HasInternet`.
4. **Encoding and Scaling** - label encoding for binary columns, ordinal encoding for
   `Contract` and `TenureGroup`, one-hot encoding for `InternetService` and `PaymentMethod`,
   and `StandardScaler` applied to continuous numeric columns.
5. **Outlier Detection and Feature Selection** - IQR and Z-score analysis (zero outliers
   found), correlation analysis, variance thresholding, and Random Forest feature importance;
   `AvgMonthlySpend` was dropped due to near-perfect collinearity with `MonthlyCharges`.
6. **Final Dataset** - 7,043 rows x 27 columns, fully numeric, zero missing values.

Full reasoning for every decision is documented inline in the notebook and in the two PDF
reports under `reports/`.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook week2_data_preprocessing.ipynb
```

## Reports

- **Business Understanding Report** - recaps the churn prediction business problem and
  connects Week 2 data quality findings back to the business objective.
- **Data Preprocessing Report** - full technical documentation of every cleaning, feature
  engineering, encoding, scaling, outlier detection, and feature selection decision.

## Professional Development

This project is accompanied by a LinkedIn post and an X (Twitter) post summarizing the
dataset, preprocessing workflow, feature engineering techniques, and key lessons learned,
tagged with #AnalystLabAfrica.
