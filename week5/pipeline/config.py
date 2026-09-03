"""
Pipeline configuration constants and paths.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "..", "week 4", "HealthConnect_Appointment_Data.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

TARGET_COL = "appointment_outcome"
POSITIVE_CLASS = "No-Show"
NEGATIVE_CLASS = "Attended"
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
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

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
}
