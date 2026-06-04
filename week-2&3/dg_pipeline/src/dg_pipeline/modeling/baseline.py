from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# I use the feature from Version 3 in the notebook, which contians Time + numeric weather + weather conditions

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


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing for the V3 baseline feature set."""
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                BASELINE_NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                BASELINE_CATEGORICAL_FEATURES,
            ),
        ]
    )

def build_linear_regression_baseline() -> Pipeline:
    """Build the notebook V3 Linear Regression baseline pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", LinearRegression()),
        ]
    )