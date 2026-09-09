"""
Data loading, cleaning, and preprocessing pipeline components.
Week 6: Added logging and validation integration.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from pipeline.config import (
    CATEGORICAL_COLS,
    DATA_PATH,
    DROP_COLS,
    NUMERICAL_COLS,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    TARGET_COL,
    logger,
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the HealthConnect appointment dataset."""
    df = pd.read_csv(path)
    logger.info("Dataset loaded: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset for modelling:
    - Filter to binary No-Show vs Attended
    - Drop irrelevant / leakage columns
    - Impute reminder_channel NMAR missingness
    - Impute distance_to_clinic_km with median
    """
    df = df.copy()

    # Binary target: keep only Attended / No-Show
    df = df[df[TARGET_COL].isin([POSITIVE_CLASS, NEGATIVE_CLASS])].reset_index(drop=True)
    logger.info("After binary filter: %d rows", df.shape[0])

    # Encode target
    df["target"] = (df[TARGET_COL] == POSITIVE_CLASS).astype(int)

    # Drop leakage and identifier columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # NMAR: missing reminder_channel means no reminder was sent
    df["reminder_channel"] = df["reminder_channel"].fillna("None_Sent")

    # MCAR: impute distance_to_clinic_km with median
    median_dist = df["distance_to_clinic_km"].median()
    df["distance_to_clinic_km"] = df["distance_to_clinic_km"].fillna(median_dist)

    # Final drop of remaining target text column
    if TARGET_COL in df.columns:
        df.drop(columns=[TARGET_COL], inplace=True)

    logger.info("After cleaning: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build a reusable sklearn ColumnTransformer that:
    - Imputes numerical features with median and applies RobustScaler
    - Imputes categorical features with most_frequent and applies OneHotEncoder
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract feature names after transformation."""
    names = []
    for name, _, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            names.extend(encoder.get_feature_names_out(cols).tolist())
    return names
