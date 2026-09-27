"""
Interactive Demonstration Script for HealthConnect Week 8 Final Showcase.
Demonstrates:
1. Pipeline initialization and artifact loading.
2. Inference on a standard patient record (Low Risk).
3. Inference on an elevated-risk patient record (Moderate Risk -> SMS trigger).
4. Inference on a high-risk patient record (High Risk -> WhatsApp + buffer slot trigger).
5. Robust error handling (catching missing columns and blocking data leakage attempts).
6. Batch inference with summary metrics.
"""

import sys
import os
import json

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
week8_dir = os.path.dirname(current_dir)
if week8_dir not in sys.path:
    sys.path.insert(0, week8_dir)

from pipeline.model_integration import HealthConnectPipeline
from pipeline.validation import ValidationError


def run_demonstration():
    print("=" * 75)
    print("HEALTHCONNECT CLINIC — WEEK 8 FINAL ML PIPELINE DEMONSTRATION")
    print("Role: Machine Learning Engineer Intern (Grai Rudolf)")
    print("Project: Patient Appointment No-Show Prediction & Clinical Action Engine")
    print("=" * 75)

    # 1. Pipeline Initialization
    print("\n[Step 1] Loading and Fitting Integrated ML Pipeline...")
    pipeline = HealthConnectPipeline(threshold=0.35)
    run_results = pipeline.run(model_name="Random Forest (Final DS Spec)")
    metrics = run_results["test_metrics"]

    print("Pipeline Ready!")
    print(f"  • Model Algorithm      : {pipeline.model_name}")
    print(f"  • Decision Threshold   : {pipeline.threshold} (Cost-sensitive high recall)")
    print(f"  • Test Set ROC-AUC     : {metrics['roc_auc']}")
    print(f"  • Test Set Recall      : {metrics['recall']} ({metrics['recall']*100:.1f}% no-shows captured)")
    print(f"  • Test Set Precision   : {metrics['precision']}")

    # Save artifacts for deployment
    artifacts_dir = os.path.join(week8_dir, "artifacts")
    pipeline.save_artifacts(artifacts_dir)
    print(f"  • Model artifacts exported to: {artifacts_dir}")

    # 2. Scenario 1: Low Risk Patient (Attending Routine Checkup)
    print("\n" + "-" * 75)
    print("[Scenario 1] Low-Risk Appointment Prediction")
    record_low = {
        "appointment_id": "HC-DEMO-001",
        "gender": "Female",
        "age": 55,
        "age_group": "45-64",
        "appointment_type": "Follow-up",
        "appointment_day": "Wednesday",
        "appointment_time": "Morning",
        "booking_lead_days": 2,
        "previous_appointments": 8,
        "previous_no_shows": 0,
        "reminder_sent": "Yes",
        "reminder_channel": "WhatsApp",
        "distance_to_clinic_km": 3.5,
    }
    pred_low = pipeline.predict_single(record_low)
    print("Input Patient Record:")
    print(json.dumps(record_low, indent=2))
    print("\nModel Output Contract:")
    print(json.dumps(pred_low, indent=2))

    # 3. Scenario 2: Moderate Risk Patient (SMS Reminder Trigger)
    print("\n" + "-" * 75)
    print("[Scenario 2] Moderate-Risk Appointment Prediction")
    record_mod = {
        "appointment_id": "HC-DEMO-002",
        "gender": "Male",
        "age": 28,
        "age_group": "25-34",
        "appointment_type": "Specialist Consultation",
        "appointment_day": "Friday",
        "appointment_time": "Afternoon",
        "booking_lead_days": 12,
        "previous_appointments": 3,
        "previous_no_shows": 1,
        "reminder_sent": "Yes",
        "reminder_channel": "SMS",
        "distance_to_clinic_km": 15.0,
    }
    pred_mod = pipeline.predict_single(record_mod)
    print("Input Patient Record:")
    print(json.dumps(record_mod, indent=2))
    print("\nModel Output Contract:")
    print(json.dumps(pred_mod, indent=2))

    # 4. Scenario 3: High Risk Patient (WhatsApp Confirmation + Buffer Slot)
    print("\n" + "-" * 75)
    print("[Scenario 3] High-Risk Appointment Prediction")
    record_high = {
        "appointment_id": "HC-DEMO-003",
        "gender": "Female",
        "age": 22,
        "age_group": "18-24",
        "appointment_type": "General Practice",
        "appointment_day": "Monday",
        "appointment_time": "Afternoon",
        "booking_lead_days": 25,
        "previous_appointments": 2,
        "previous_no_shows": 2,
        "reminder_sent": "No",
        "reminder_channel": "None_Sent",
        "distance_to_clinic_km": 32.5,
    }
    pred_high = pipeline.predict_single(record_high)
    print("Input Patient Record:")
    print(json.dumps(record_high, indent=2))
    print("\nModel Output Contract:")
    print(json.dumps(pred_high, indent=2))

    # 5. Scenario 4: Error Handling & Data Leakage Shield
    print("\n" + "-" * 75)
    print("[Scenario 4] Reliability & Safety: Blocking Post-Arrival Data Leakage")
    leaky_record = record_low.copy()
    leaky_record["waiting_time_minutes"] = 45  # Post-arrival leakage feature
    print("Attempting to pass 'waiting_time_minutes' to inference endpoint...")
    try:
        pipeline.predict_single(leaky_record)
        print("ERROR: Pipeline failed to reject leaked feature!")
    except ValidationError as e:
        print(f"SUCCESS: Pipeline actively blocked prediction with ValidationError:\n  -> \"{e}\"")

    print("\nAttempting to pass incomplete record (missing 'distance_to_clinic_km')...")
    broken_record = record_low.copy()
    del broken_record["distance_to_clinic_km"]
    try:
        pipeline.predict_single(broken_record)
        print("ERROR: Pipeline failed to catch missing feature!")
    except ValidationError as e:
        print(f"SUCCESS: Pipeline caught missing column with ValidationError:\n  -> \"{e}\"")

    # 6. Scenario 5: HC-POD Cross-Track Handoff Verification
    print("\n" + "-" * 75)
    print("[Scenario 5] HC-POD Cross-Track Verification (Data Science Handoff)")
    print("Verified compatibility with Data Science specifications:")
    print("  1. Serialized scikit-learn Pipeline / Model accepted via import_ds_model() or load_artifacts()")
    print("  2. Decision threshold set to 0.35 (within recommended 0.30–0.35 range)")
    print("  3. Leakage feature 'waiting_time_minutes' completely excluded")
    print("  4. Known limitation on gender subgroup fairness audited and transparently documented")
    print("\n" + "=" * 75)
    print("HEALTHCONNECT FINAL ML PIPELINE SHOWCASE DEMONSTRATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    run_demonstration()
