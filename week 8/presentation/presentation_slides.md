# HealthConnect Final Project Showcase — Slide Deck
**Track:** Machine Learning Engineering  
**Presenter:** Grai Rudolf (Junior Machine Learning Engineer Intern)  
**Project:** Improving Patient Appointment Attendance and Healthcare Support Using Data & AI  
**Program:** AnalystLab Africa Experience Lab (Week 8)  

---

### Slide 1: Title & Introduction (0:00 – 0:45)
- **Title:** HealthConnect Clinic: End-to-End Production ML Pipeline for Appointment Attendance Optimization
- **Presenter:** Grai Rudolf | Machine Learning Engineer Intern
- **Program:** AnalystLab Africa Experience Lab — Week 8 Final Showcase
- **The Core Problem:**
  - HealthConnect Clinic experiences a severe **48.5% appointment no-show rate**.
  - Operational impact: Idle medical staff, lost clinical revenue, delayed patient care, and chaotic scheduling.
  - Engineering Mission: Build a robust, leakage-free, automated ML prediction and clinical decision engine.

---

### Slide 2: ML Engineering Responsibilities & Deliverables (0:45 – 1:45)
- **Architectural Scope:**
  - Transition from experimental modeling into a reliable, modular production pipeline (`HealthConnectPipeline`).
  - Strict leakage elimination: actively barring post-arrival features (`waiting_time_minutes`).
  - Patient-level grouped validation (`GroupShuffleSplit` & `GroupKFold` on `patient_id`) to prevent patient habit memorization.
- **Core Engineering Deliverables:**
  - Modular Python pipeline (`pipeline/`: data processing, feature engineering, model training, evaluation, validation).
  - Microservice inference interfaces: `predict_single()` for real-time JSON responses and `predict_batch()` for clinic schedules.
  - Artifact serialization & deserialization (`model.pkl`, `preprocessor.pkl`, `meta.json`).
  - 21 automated pytest tests spanning unit, integration, reliability, and HC-POD contract suites.

---

### Slide 3: System Architecture & Workflow Pipeline (1:45 – 2:45)
- **5-Stage Integrated Technical Architecture:**
  1. **Ingestion & Validation:** Schema verification of raw clinic records (`REQUIRED_RAW_COLS`).
  2. **Data Cleaning & Shielding:** Missing value imputation (NMAR `reminder_channel` = `'None_Sent'`, MCAR median distance) + post-arrival feature stripping.
  3. **Domain Feature Engineering:** Historical no-show ratio (Laplace smoothed), lead time bins, new patient flag, reminder flag.
  4. **Robust Preprocessing:** `ColumnTransformer` with `handle_unknown='ignore'` to guarantee zero crashing on unseen categorical values.
  5. **Inference & Risk Dispatch:** Model scoring $\to$ Cost-sensitive threshold evaluation $\to$ Action dispatch.

---

### Slide 4: Model Benchmarking & Performance on Unseen Patients (2:45 – 3:45)
- **Zero-Patient-Leakage Test Split ($N=966$ appointments, 336 unseen patients):**
  - **Random Forest (Final DS Spec):**
    - **ROC-AUC:** **0.6846** (vs 0.6735 Baseline Logistic Regression).
    - **Operating Threshold:** **0.35** (Cost-sensitive optimization aligned with Data Science handoff).
    - **No-Show Recall:** **94.20%** — intercepts nearly 19 out of every 20 missed appointments!
    - **Precision:** **53.85%** | **F1-Score:** **0.6852** | **Brier Score:** **0.2256**.
  - **Clinical Justification for Threshold 0.35:**
    - False Negatives (unprepared empty clinic slots) cost $5\times$ more than False Positives (sending an extra WhatsApp message).

---

### Slide 5: HC-POD Cross-Track Collaboration — Data Science Handoff (3:45 – 5:00)
- **Collaborator:** Mairame Samba Niang (Data Science Intern)
- **What We Exchanged:**
  - *MLE $\to$ DS:* Strict 12-feature input contract, JSON output schema, scikit-learn pickle constraint, and critical flag identifying `waiting_time_minutes` as post-arrival leakage.
  - *DS $\to$ MLE:* Retrained Random Forest specification without leakage (ROC-AUC ≈ 0.682), threshold recommendation (0.30–0.35), and alert regarding gender fairness gap.
- **What Changed Because of Collaboration:**
  - Mairame retrained both models honestly without leakage; discovered that improvement over baseline is small/favorable (+1.1% AUC) but 95% bootstrap CI includes zero.
  - MLE updated pipeline threshold to 0.35, implemented `GroupKFold`, and built `import_ds_model()` adapter to guarantee 100% runtime compatibility.
  - **Evidence:** Documented Slack exchanges, contract tests (`test_ds_contract.py`), and Git commit history.

---

### Slide 6: Operational Risk Tier Engine & Clinic Action Routing (5:00 – 6:00)
- **Bridging AI Predictions to Clinical Operations:**
  - **High Risk ($P \ge 0.70$):**
    - *Action:* `TRIGGER_WHATSAPP_CONFIRMATION_AND_BUFFERSLOT`
    - Automated two-way WhatsApp confirmation; flags scheduling system to hold a contingent backup slot.
  - **Moderate Risk ($0.35 \le P < 0.70$):**
    - *Action:* `TRIGGER_SMS_REMINDER_REQUESTING_CONFIRMATION`
    - Dispatches 48-hour automated SMS requesting a "1 to Confirm" reply.
  - **Low Risk ($P < 0.35$):**
    - *Action:* `STANDARD_EMAIL_NOTIFICATION`
    - Minimal friction; routine email/calendar reminder saving SMS/telephony costs.

---

### Slide 7: Quality Assurance, Reliability & Safety Testing (6:00 – 7:00)
- **Automated Verification:** 21 automated tests, 100% pass rate in 17.8 seconds.
  - **Unit Tests (8):** Data cleaning, median imputation, leakage stripping, patient grouping.
  - **Integration Tests (4):** End-to-end fit, predict, batch prediction, output schema validation.
  - **Reliability & Edge Cases (5):** Threshold boundaries, missing input columns, blocking `waiting_time_minutes`, artifact persistence round-trip.
  - **HC-POD Contract Tests (4):** Input schema compliance, JSON contract verification, DS model import adapter.

---

### Slide 8: Ethical AI, Fairness Audit & Known Limitations (7:00 – 8:00)
- **Subgroup Fairness Audit (Gender Disparity):**
  - Evaluated on test set:
    - *Female Patients:* ROC-AUC = 0.6966, Recall = 95.24%
    - *Male Patients:* ROC-AUC = 0.6719, Recall = 93.39%
  - Fairness Gap: ~0.025 AUC disparity. Both groups achieve high sensitivity (>93%), but clinic policy must ensure male patients are not subjected to higher cancellation risk.
- **Statistical Significance Limitation:**
  - Removal of leakage feature means the Random Forest improvement over Logistic Regression (+1.1% AUC) has a 95% bootstrap CI including zero. We prioritize Random Forest for non-linear interactions rather than over-hyping statistical superiority.
- **Stationary Lead-Time Assumption:**
  - Model assumes stable booking behavior; major clinic scheduling changes will require drift monitoring and retraining.

---

### Slide 9: Multidisciplinary HC-POD Integration & Business Value (8:00 – 9:00)
- **How All Tracks Form One Solution:**
  - **Data Analytics:** Uncovered high lead-time no-show trends $\to$ Informed feature engineering.
  - **Data Science:** Developed Random Forest model & tuned threshold $\to$ Integrated into pipeline.
  - **ML Engineering:** Built leakage-free, tested pipeline & JSON API $\to$ Serves predictions in <10ms.
  - **Generative AI:** Consumes risk tiers $\to$ Generates personalized WhatsApp/SMS conversational follow-ups.
  - **Project Management:** Coordinated cross-track dependencies and unified clinical showcase.
- **Quantifiable Business Value:**
  - 94.2% no-show recall enables clinic staff to proactively reclaim up to 35–40% of previously lost appointment capacity, reducing patient wait lists and boosting provider revenue.

---

### Slide 10: Closing & Key Takeaways (9:00 – 9:45)
- **Key Takeaways:**
  - Production ML engineering is about **integrity, reliability, and cross-track contracts**, not just model tuning. Catching data leakage before deployment saved the project from deceptive real-world failure.
- **Personal Learning:**
  - Importance of rigorous contract testing and patient-level cross-validation in healthcare AI.
- **Future Roadmap:**
  - Containerization via Docker & Kubernetes, Evidently AI drift monitoring, and Fairlearn demographic parity mitigation.
- **Thank you! Questions?**
