from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from dg_pipeline.utils.feature_config import FEATURE_SETS

import pandas as pd

DEFAULT_SPLIT_RATIO = 0.8
DEFAULT_RANDOM_STATE = 42

def random_train_test_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a random train/test split from model-ready data.
    """

    data = data.copy()
    data["hour"] = pd.to_datetime(data["hour"])
    data = data.sort_values("hour").reset_index(drop=True)

    train_data, test_data = train_test_split(
        data,
        train_size=DEFAULT_SPLIT_RATIO,
        random_state=DEFAULT_RANDOM_STATE,
        shuffle=True,
    )

    train_data = train_data.sort_values("hour").reset_index(drop=True)
    test_data = test_data.sort_values("hour").reset_index(drop=True)

    return train_data, test_data



def time_based_train_test_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a chronological train/test split from model-ready data.
    """

    data = data.copy()
    data["hour"] = pd.to_datetime(data["hour"])
    data = data.sort_values("hour").reset_index(drop=True)

    split_index = int(len(data) * DEFAULT_SPLIT_RATIO)

    train_data = data.iloc[:split_index].copy().reset_index(drop=True)
    test_data = data.iloc[split_index:].copy().reset_index(drop=True)

    return train_data, test_data



def train_test_split_by_strategy(
        data: pd.DataFrame,
    feature_config: dict,
    split_strategy: str = "chronological",
):
    """
    Create train/test split based on the selected strategy.

    Supported strategies:
    - chronological
    - random
    """

    if split_strategy == "chronological":
        return time_based_train_test_split(
            data=data,
        )

    if split_strategy == "random":
        return random_train_test_split(
            data=data,
        )

    raise ValueError(
        f"Unknown split_strategy: {split_strategy}. "
        "Expected 'chronological' or 'random'."
    )




def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build preprocessing pipeline for numeric and categorical features.
    """

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
        ]
    )


def build_model_pipeline(
    regressor,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Build a full sklearn pipeline with preprocessing and a regressor.
    """

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                ),
            ),
            ("regressor", regressor),
        ]
    )


def build_linear_regression_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    return build_model_pipeline(
        regressor=LinearRegression(),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_random_forest_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    return build_model_pipeline(
        regressor=RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_gradient_boosting_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    return build_model_pipeline(
        regressor=GradientBoostingRegressor(
            random_state=42,
        ),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )



# def to_series(data) -> pd.Series:
#     """Convert a one-column DataFrame or Series to Series."""
#     if isinstance(data, pd.Series):
#         return data

#     if isinstance(data, pd.DataFrame):
#         if data.shape[1] != 1:
#             raise ValueError(
#                 "Expected a one-column DataFrame when converting to Series, "
#                 f"but got {data.shape[1]} columns."
#             )
#         return data.iloc[:, 0]

#     raise TypeError(f"Expected Series or DataFrame, got {type(data)}.")



XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

TUNED_XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

def build_xgboost_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Build XGBoost model with engineered features."""
    return build_model_pipeline(
        regressor=XGBRegressor(**XGBOOST_PARAMS),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_tuned_xgboost_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Build tuned XGBoost model with engineered features."""
    return build_model_pipeline(
        regressor=XGBRegressor(**TUNED_XGBOOST_PARAMS),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


