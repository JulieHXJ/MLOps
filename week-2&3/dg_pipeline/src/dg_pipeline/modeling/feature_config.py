from dagster import Definitions, definitions

from dg_pipeline.io_managers.csv_io_manager import CsvIOManager
from dg_pipeline.resources.data_loader import BikeRentalDataLoader





BASELINE_TARGET = "total_count"

BASELINE_NUMERIC_FEATURES = [
    "temperature_c",
    "humidity",
    "windspeed_kmh",
    "is_holiday",
]


BASELINE_CATEGORICAL_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "month",
    "conditions",
]

BASELINE_FEATURES = (
    BASELINE_NUMERIC_FEATURES
    + BASELINE_CATEGORICAL_FEATURES
)




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



FEATURE_SETS = {
    "baseline": {
        "target": BASELINE_TARGET,
        "numeric_features": BASELINE_NUMERIC_FEATURES,
        "categorical_features": BASELINE_CATEGORICAL_FEATURES,
        "features": BASELINE_FEATURES,
        "description": "Baseline: weather + holiday + basic time features",
    },
    "engineered": {
        "target": ENGINEERED_TARGET,
        "numeric_features": ENGINEERED_NUMERIC_FEATURES,
        "categorical_features": ENGINEERED_CATEGORICAL_FEATURES,
        "features": ENGINEERED_FEATURES,
        "description": "Engineered: baseline + calendar and historical features",
    },
}
