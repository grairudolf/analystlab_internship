# AnalystLab Africa — Machine Learning Internship Programme

## Week 1: Business Understanding and Exploratory Data Analysis (EDA)

**Intern:** Rudolf  
**Business Scenario:** ABC Communications Ltd — Customer Churn Prediction  

---

## Project Overview

This folder contains the Week 1 deliverable for the AnalystLab Africa Machine Learning Internship Programme. The objective of Week 1 is to establish the business context, frame the customer churn problem as a binary classification task, and perform comprehensive Exploratory Data Analysis (EDA) on the Telco Customer Churn dataset.

---

## Folder Structure

```
week 1/
├── README.md                          # Week 1 Overview & Documentation
├── week1_churn_analysis.ipynb         # EDA & Business Understanding Jupyter Notebook
└── WA_Fn-UseC_-Telco-Customer-Churn.csv # Original Raw Telco Dataset
```

---

## Key Findings from Exploratory Data Analysis

1. **Overall Target Class Distribution:**
   - **Retained Customers (No):** 5,174 (73.46%)
   - **Churned Customers (Yes):** 1,869 (26.54%)
   - Class imbalance ratio is approximately **2.77 to 1**.

2. **Contract Type & Churn Risk:**
   - **Month-to-Month Contracts:** Highest churn rate at **42.71%**.
   - **One-Year Contracts:** Low churn rate at **11.27%**.
   - **Two-Year Contracts:** Lowest churn rate at **2.83%**.

3. **Internet Service Type:**
   - **Fiber Optic Subscribers:** High churn rate at **41.89%** (driven by higher monthly charges $\sim \$90+$ and service expectations).
   - **DSL Subscribers:** Moderate churn rate at **18.96%**.
   - **No Internet Service:** Lowest churn rate at **7.40%**.

4. **Customer Tenure Lifecycle:**
   - The highest concentration of churn occurs during the **first 12 months** of customer tenure before brand loyalty is established.

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook week1_churn_analysis.ipynb
```
