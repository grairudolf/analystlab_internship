"""
Pipeline configuration constants and paths.
Week 6: Added logging config, validation schema, model registry.
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

# ── Train / test ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ── Model registry (Week 6) ─────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.50

# ── Validation schema (Week 6) ──────────────────────────────────────────────
REQUIRED_RAW_COLS = {
    "appointment_id", "patient_id", "gender", "age", "age_group",
    "appointment_type", "booking_date", "appointment_date",
    "appointment_day", "appointment_time", "booking_lead_days",
    "previous_appointments", "previous_no_shows", "reminder_sent",
    "reminder_channel", "distance_to_clinic_km", "waiting_time_minutes",
    "appointment_outcome",
}

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
    "default_threshold": DEFAULT_THRESHOLD,
}
