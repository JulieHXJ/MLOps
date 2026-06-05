from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import pandas as pd


def create_train_test_split(
    data: pd.DataFrame,
    feature_config: dict,
    split_ratio: float = 0.8,
):
    """
    Create a chronological train/test split from model-ready data.
    """

    data = data.copy()
    data["hour"] = pd.to_datetime(data["hour"])
    data = data.sort_values("hour").reset_index(drop=True)

    features = feature_config["features"]
    target = feature_config["target"]

    split_index = int(len(data) * split_ratio)

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    X_train = train_data[features]
    X_test = test_data[features]
    y_train = train_data[target]
    y_test = test_data[target]
    test_hours = test_data["hour"]

    return X_train, X_test, y_train, y_test, test_hours, train_data, test_data





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