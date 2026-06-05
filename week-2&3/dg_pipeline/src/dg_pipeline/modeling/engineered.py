import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ENGINEERED_TARGET = "total_count"

ENGINEERED_NUMERIC_FEATURES = [
    # weather
    "temperature_c",
    "perceived_temperature_c",
    "humidity",
    "windspeed_kmh",

    # calendar
    "is_weekend",
    "is_holiday",
    "is_workday",
    "rush_hour",
    "was_missing_hour",

    # cyclical time features
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",

    # historical demand features
    "total_lag_24h",
    "total_lag_7d",
    "total_24h_mean",
    "total_7d_mean",
    "same_hour_weekday_4w_mean",
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "conditions",
    "season",
]

ENGINEERED_FEATURES = ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES


def build_engineered_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipeline for engineered features."""
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                ENGINEERED_NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                ENGINEERED_CATEGORICAL_FEATURES,
            ),
        ]
    )

def build_engineered_linear_regression() -> Pipeline:
    """Build Linear Regression pipeline with engineered features."""
    return Pipeline(
        steps=[
            ("preprocessor", build_engineered_preprocessor()),
            ("regressor", LinearRegression()),
        ]
    )


def build_engineered_random_forest() -> Pipeline:
    """Build Random Forest pipeline with engineered features."""
    return Pipeline(
        steps=[
            ("preprocessor", build_engineered_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_engineered_gradient_boosting() -> Pipeline:
    """Build Gradient Boosting pipeline with engineered features."""
    return Pipeline(
        steps=[
            ("preprocessor", build_engineered_preprocessor()),
            (
                "regressor",
                GradientBoostingRegressor(
                    random_state=42,
                ),
            ),
        ]
    )

