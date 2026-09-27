# HealthConnect Final Project Showcase — Spoken Video Presentation Script
**Speaker:** Grai Rudolf  
**Role:** Junior Machine Learning Engineer Intern  
**Track:** Machine Learning Engineering Track  
**Project:** HealthConnect Clinic Experience Lab  
**Target Duration:** ~8 minutes (Within the mandatory 5–10 minute window)  

---

### [0:00 – 0:45] SECTION 1: INTRODUCTION & PROJECT CONTEXT
*(Display Slide 1 — Title & Introduction)*

> "Hello everyone, evaluators, mentors, and fellow interns. My name is **Grai Rudolf**, and I am serving as a **Junior Machine Learning Engineer Intern** in the AnalystLab Africa Experience Lab. 
> 
> Today, I am proud to present our final Week 8 integration and project showcase for **HealthConnect Clinic**. 
> 
> HealthConnect Clinic is a multi-specialty healthcare provider facing a severe operational bottleneck: **48.5% of scheduled patient appointments result in no-shows**. This staggering missed-appointment rate causes wasted clinical hours, lost provider revenue, and dangerously delays essential care for patients who need it most. 
> 
> Our mission throughout this 5-week lab was clear: how can HealthConnect use data, predictive modeling, and artificial intelligence to proactively anticipate no-shows, optimize clinic capacity, and provide automated patient support?"

---

### [0:45 – 2:30] SECTION 2: MY TRACK CONTRIBUTION & THE ML WORKFLOW
*(Display Slide 2 & Slide 3 — Track Contribution & System Architecture)*

> "As the Machine Learning Engineer for our HC-POD, my core responsibility was taking the exploratory analyses and experimental models from earlier weeks and hardening them into an **end-to-end, production-grade, reproducible ML pipeline**. 
> 
> An algorithm in a notebook cannot serve a clinic. It needs a structured engineering backbone that ingests raw data, handles missing values, prevents data leakage, executes predictions in real time, and persists trained artifacts reliably.
> 
> To accomplish this, I developed the modular `HealthConnectPipeline` package, which is structured into five core stages:
> 
> First, **Ingestion and Strict Validation**. We validate raw input schemas against our documented clinic contract, verifying all 18 fields.
> 
> Second, **Data Cleaning and Leakage Shielding**. In healthcare, data leakage is fatal. We actively strip post-arrival features such as `waiting_time_minutes`. Because waiting time is only recorded after a patient arrives at the clinic, using it to predict attendance is impossible in production. Our cleaning module actively deletes it and our inference validator blocks it with custom `ValidationError` exceptions if an external service tries to pass it.
> 
> Third, **Patient-Level Grouped Splitting**. Because our 5,000 appointment records represent 1,680 unique patients who visit repeatedly, random splitting causes patient leakage—the model memorizes individual patient behavior. I implemented `GroupShuffleSplit` on `patient_id`, guaranteeing that our test set of 966 appointments contains 336 patients who are completely unseen by the training algorithm.
> 
> Fourth, **Domain Feature Engineering and Preprocessing**. We engineered Laplace-smoothed historical no-show ratios, booking lead-time buckets, and new patient flags. Our preprocessing pipeline utilizes `ColumnTransformer` with `handle_unknown='ignore'` on categorical one-hot encoders, ensuring that if a patient appears with an unexpected category in production, the system handles it gracefully without crashing.
> 
> Fifth, **Artifact Persistence**. We implemented full round-trip serialization with `save_artifacts()` and `load_artifacts()`, storing the fitted preprocessor, model binary, and metadata schema."

---

### [2:30 – 4:00] SECTION 3: MODEL PERFORMANCE & OPERATING THRESHOLD
*(Display Slide 4 — Model Benchmarking & Performance)*

> "Now, let’s examine our model performance evaluated strictly on unseen patients. 
> 
> We benchmarked our primary candidate—a tuned **Random Forest Classifier**—against a baseline Logistic Regression. 
> 
> On our unseen test set, the Random Forest achieved an **ROC-AUC of 0.6846**, outperforming the baseline AUC of 0.6735. 
> 
> However, in a clinical environment, the standard default decision threshold of 0.50 is economically and operationally suboptimal. If a clinic misses a no-show—a False Negative—a doctor sits idle for 30 minutes, costing the clinic significant revenue. If the clinic falsely predicts a no-show—a False Positive—the only cost is sending an automated WhatsApp confirmation message. 
> 
> Therefore, in close alignment with our Data Science track, we calibrated our operating decision threshold to **0.35**. 
> 
> At a 0.35 threshold, our pipeline achieves an outstanding **Recall of 94.20%**. This means that out of every 20 patients who will miss their appointment, our system proactively identifies and intercepts 19 of them ahead of time! Our Precision remains steady at **53.85%**, providing the ideal operational balance between high sensitivity and manageable outreach volume."

---

### [4:00 – 5:45] SECTION 4: HC-POD CROSS-TRACK COLLABORATION (DATA SCIENCE HANDOFF)
*(Display Slide 5 — HC-POD Cross-Track Collaboration)*

> "One of the most defining aspects of Week 8 was our **HC-POD cross-track collaboration**, specifically between myself in ML Engineering and **Mairame Samba Niang in Data Science**. 
> 
> True collaboration isn't just sending messages—it is the active, reciprocal exchange of deliverables and technical constraints. 
> 
> Earlier this week, Mairame reached out with her finalized Week 7 candidate model, asking what input and output formats my pipeline required and what technical constraints existed. 
> 
> I provided her with our exact 12-feature input schema, our JSON output specification, and two critical technical constraints: first, that serialization must be pickle-compatible, and second, that `waiting_time_minutes` must be completely omitted due to post-arrival data leakage. 
> 
> Mairame immediately retrained her model without `waiting_time_minutes` on the real dataset and re-verified it. She confirmed that removing the leakage feature did not hurt genuine performance—her ROC-AUC remained virtually unchanged at 0.682. 
> 
> Crucially, she also demonstrated exceptional scientific integrity: she discovered that after removing the leakage feature from both models, the improvement of Random Forest over the baseline was no longer statistically significant at 95% confidence, as the bootstrap confidence interval included zero. She transparently revised her claim to a 'small, favorable, but non-conclusive' gain. 
> 
> On my side, I built a dedicated Data Science adapter method, `import_ds_model()`, in our pipeline, updated our decision threshold to her recommended 0.35, and wrote contract tests in `test_ds_contract.py` that automatically verify that her model specifications and our pipeline interfaces integrate with 100% compatibility."

---

### [5:45 – 7:00] SECTION 5: CLINICAL RISK ENGINE, TESTING & LIMITATIONS
*(Display Slide 6, Slide 7 & Slide 8 — Risk Tiers, QA & Limitations)*

> "To make these predictions usable for clinic receptionists and automated communication systems, we built an **Actionable Risk-Tier Decision Engine**:
> 
> - Patients with a predicted probability of 70% or higher are classified as **HIGH_RISK**. The system outputs the action `TRIGGER_WHATSAPP_CONFIRMATION_AND_BUFFERSLOT`. This alerts the clinic scheduling system to prepare a contingency buffer slot and dispatches an interactive WhatsApp confirmation bot.
> - Patients between 35% and 70% probability are tagged as **MODERATE_RISK**, triggering an automated SMS reminder requesting a reply confirmation.
> - Patients under 35% are **LOW_RISK**, receiving standard email reminders, saving SMS telephony costs.
> 
> To ensure production reliability, I implemented a comprehensive QA suite with **21 automated pytest tests** covering data validation, leakage prevention, patient grouping, pipeline integration, error handling, artifact round-trips, and cross-track contracts. All 21 tests pass with a 100% success rate in under 18 seconds.
> 
> In accordance with healthcare AI ethics, we must also be fully transparent about our **known limitations**:
> First, our demographic fairness audit revealed a slight gender disparity: Female patients achieve an ROC-AUC of 0.6966 compared to 0.6719 for Male patients. While both genders achieve high recall over 93%, clinic operations must ensure automated cancellation policies do not unfairly penalize male patients.
> Second, as Mairame discovered, the small margin of improvement over the baseline means we rely on Random Forest primarily for its ability to model non-linear interactions and lead-time bin thresholds, rather than claiming statistical superiority."

---

### [7:00 – 8:30] SECTION 6: OVERALL SOLUTION, BUSINESS VALUE & CLOSING
*(Display Slide 9 & Slide 10 — Overall Solution, Business Value & Closing)*

> "Finally, let's look at how all the HC-POD tracks unite into a single, cohesive HealthConnect solution. 
> 
> **Data Analytics** provided the foundational insights on patient lead times and missing reminder data. **Data Science** translated those patterns into our Random Forest model and cost-sensitive threshold. **ML Engineering** built the leakage-free, tested pipeline and API contract. **Generative AI** consumes our risk-tier outputs to generate tailored, compassionate patient messages. And **Project Management** unified our timelines and deliverables.
> 
> By intercepting 94.2% of no-shows and routing high-risk slots into buffer allocations, HealthConnect Clinic can reclaim up to **35% to 40% of previously lost clinic appointment capacity**, recovering thousands of dollars in monthly revenue while ensuring patients receive timely medical care.
> 
> **My Key Takeaways:**
> Working on HealthConnect taught me that Machine Learning Engineering is fundamentally about **integrity, system robustness, and cross-disciplinary communication**. The most sophisticated model is useless if it leaks data, fails on unseen categories, or cannot communicate with downstream systems. Catching data leakage and establishing clear API contracts with Data Science transformed this project into a real-world, dependable solution.
> 
> For future iterations, I would implement Docker containerization, live data drift monitoring using Evidently AI, and algorithmic re-weighting to completely close the gender fairness gap.
> 
> Thank you to the AnalystLab Africa team, our mentors, and my HC-POD team members for an incredible learning experience. I look forward to your questions."

---

### Tips for Recording Your Presentation:
1. **Software:** Use Loom, Zoom (record to local computer), or OBS Studio.
2. **Camera:** Make sure your webcam is turned on and visible in the corner while sharing the slides on screen.
3. **Pacing:** Speak naturally and clearly. The script is structured to comfortably take between 7.5 and 8.5 minutes.
4. **Visuals:** You can open `week 8/presentation/presentation_slides.md` or the compiled PDF reports (`week8_final_ml_pipeline_integration_report.pdf`) and scroll through the figures as you speak.
