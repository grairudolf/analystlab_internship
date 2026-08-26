# Week 4: HealthConnect Experience Lab (Machine Learning Track)

**Author:** Rudolf  
**Role:** Machine Learning Engineer Intern  
**Project:** HealthConnect Clinic — Appointment No-Show Prediction  
**Date:** August 2026  

---

## Overview

In Week 4, I began work on the **HealthConnect Experience Lab**. HealthConnect Clinic experiences high patient appointment no-shows (~48.5% baseline non-attendance rate across 5,000 records). 

My goal for this week was to analyze the dataset, define the machine learning problem, design a production ML system architecture, and generate technical documentation and LaTeX reports ahead of model training in Week 5.

---

## Project Structure

```
week 4/
├── AnalystLab_Africa_Week4_Experience_Lab_Assignment.pdf    # Assignment details & guidelines
├── HealthConnect_Appointment_Data.csv                       # Raw dataset (5,000 appointments)
├── HealthConnect_Clinic_Knowledge_Base.docx                 # Operational rules & policies
├── HealthConnect_Data_Dictionary.xlsx                       # Data schema & column definitions
├── README.md                                                 # This file
├── week4_healthconnect_ml_definition_system_design.ipynb   # Main Jupyter analysis notebook
├── figures/                                                 # Generated diagnostic figures
│   ├── fig01_outcome_distribution.png
│   ├── fig02_lead_days_vs_noshow.png
│   ├── fig03_previous_noshows_impact.png
│   ├── fig04_reminder_channel_attendance.png
│   └── fig05_system_architecture.png
└── reports/                                                 # LaTeX reports & compiled PDFs
    ├── week4_ml_problem_definition_and_system_design.tex
    ├── week4_ml_problem_definition_and_system_design.pdf
    ├── week4_project_summary.tex
    └── week4_project_summary.pdf
```

---

## Key Technical Decisions & Data Observations

1. **Target Formulation:**
   - Evaluated 5,000 records: `No-Show` (48.46%), `Attended` (46.28%), and `Cancelled` (5.26%).
   - Filtered out `Cancelled` appointments during binary classification modeling because cancellations represent advance notice that allows the clinic to fill open slots. The binary model focuses specifically on predicting `No-Show` (1) vs `Attended` (0).

2. **Preventing Data Leakage:**
   - Identified `waiting_time_minutes` as a post-arrival metric. It is explicitly excluded from feature matrices used for pre-appointment prediction.

3. **Handling Missing Values:**
   - `reminder_channel` (27.3% missing): Missing Not At Random (NMAR). Missing values mean no reminder was sent. Categorized as `"None_Sent"`.
   - `distance_to_clinic_km` (1.8% missing): Missing Completely At Random (MCAR). Imputed using median values grouped by postal region.

4. **Cross-Validation Strategy:**
   - 1,696 unique patients make up the 5,000 appointment records. To avoid data leakage across repeat patient visits, validation must use `GroupKFold` on `patient_id` or a temporal split.

---

## System Architecture

Designed a 5-stage production pipeline for real-time inference at $T-48$ hours before scheduled appointments:
1. **Ingestion:** Pulls upcoming appointment data from the EHR database.
2. **Feature Store:** Imputes missing values, calculates historical patient rates, and validates schemas.
3. **Inference API:** FastAPI container serving model predictions ($P(\text{No-Show})$).
4. **Decision Engine:** Stratifies risk tiers (High > 70%, Moderate 40-70%, Low < 40%) to trigger automated WhatsApp/SMS reminders and standby buffer bookings.
5. **Monitoring:** Logs inputs and predictions to track feature and concept drift using Evidently AI and MLflow.

---

## How to Run

To execute the notebook and reproduce figures:

```bash
# Open notebook in Jupyter
jupyter notebook week4_healthconnect_ml_definition_system_design.ipynb
```
