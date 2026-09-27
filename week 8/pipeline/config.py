"""
Pipeline configuration constants, paths, and validation schemas for Week 8.
Week 8: Final integration configuration supporting patient-level grouping (GroupKFold),
Data Science model handoff threshold (0.35), risk tier mappings, and operational actions.
"""

import os
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "..", "week 4", "HealthConnect_Appointment_Data.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger("healthconnect.pipeline")

# ── Target ───────────────────────────────────────────────────────────────────
TARGET_COL = "appointment_outcome"
POSITIVE_CLASS = "No-Show"
NEGATIVE_CLASS = "Attended"

# ── Column definitions ───────────────────────────────────────────────────────
# Explicit leakage prevention: waiting_time_minutes must never enter the model
DROP_COLS = [
    "appointment_id",
    "patient_id",
    "waiting_time_minutes",
    "appointment_outcome",
    "booking_date",
    "appointment_date",
]

CATEGORICAL_COLS = [
    "gender",
    "age_group",
    "appointment_type",
    "appointment_day",
    "appointment_time",
    "reminder_sent",
    "reminder_channel",
]

NUMERICAL_COLS = [
    "age",
    "booking_lead_days",
    "previous_appointments",
    "previous_no_shows",
    "distance_to_clinic_km",
]

ENGINEERED_NUM_COLS = [
    "historical_noshow_ratio",
    "is_new_patient",
    "has_reminder",
]

ENGINEERED_CAT_COLS = [
    "lead_time_bin",
]

# ── Train / Test & Validation Settings ───────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
GROUP_COL = "patient_id"

# ── Model Registry & Thresholding (DS Alignment) ────────────────────────────
# Aligned with Data Science handoff from Mairame Samba Niang (ROC-AUC 0.682)
DEFAULT_THRESHOLD = 0.35
RECOMMENDED_THRESHOLD_RANGE = (0.30, 0.35)

# ── Operational Risk Tiers & Clinical Actions ───────────────────────────────
RISK_TIERS = {
    "HIGH": {
        "threshold": 0.70,
        "label": "HIGH_RISK",
        "action": "TRIGGER_WHATSAPP_CONFIRMATION_AND_BUFFERSLOT",
    },
    "MODERATE": {
        "threshold": 0.35,
        "label": "MODERATE_RISK",
        "action": "TRIGGER_SMS_REMINDER_REQUESTING_CONFIRMATION",
    },
    "LOW": {
        "threshold": 0.0,
        "label": "LOW_RISK",
        "action": "STANDARD_EMAIL_NOTIFICATION",
    },
}

# ── Validation Schemas ──────────────────────────────────────────────────────
REQUIRED_RAW_COLS = {
    "appointment_id", "patient_id", "gender", "age", "age_group",
    "appointment_type", "booking_date", "appointment_date",
    "appointment_day", "appointment_time", "booking_lead_days",
    "previous_appointments", "previous_no_shows", "reminder_sent",
    "reminder_channel", "distance_to_clinic_km", "waiting_time_minutes",
    "appointment_outcome",
}

# Prediction input contract shared with Data Science & API callers
PREDICTION_INPUT_COLS = [
    "gender",
    "age",
    "age_group",
    "appointment_type",
    "appointment_day",
    "appointment_time",
    "booking_lead_days",
    "previous_appointments",
    "previous_no_shows",
    "reminder_sent",
    "reminder_channel",
    "distance_to_clinic_km",
]

PIPELINE_CONFIG = {
    "data_path": DATA_PATH,
    "target_col": TARGET_COL,
    "positive_class": POSITIVE_CLASS,
    "negative_class": NEGATIVE_CLASS,
    "drop_cols": DROP_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "numerical_cols": NUMERICAL_COLS,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "cv_folds": CV_FOLDS,
    "group_col": GROUP_COL,
    "default_threshold": DEFAULT_THRESHOLD,
}
