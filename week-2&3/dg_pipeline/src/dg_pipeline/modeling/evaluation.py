from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REQUIRED_PREDICTION_COLUMNS = [
    "hour",
    "actual_total_count",
    "predicted_total_count",
]


def prepare_train_test_split(
    model_ready_data: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split 80/20 use for final model comparison."""
    required_columns = [
        "hour",
        *features,
        target,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in model_ready_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Model cannot be trained because required columns "
            f"are missing: {missing_columns}"
        )

    model_data = model_ready_data[required_columns].copy()
    model_data["hour"] = pd.to_datetime(model_data["hour"])

    model_data = (
        model_data
        .sort_values("hour")
        .reset_index(drop=True)
    )

    split_index = int(len(model_data) * 0.8)

    train_data = model_data.iloc[:split_index].copy()
    test_data = model_data.iloc[split_index:].copy()

    X_train = train_data[features]
    X_test = test_data[features]
    y_train = train_data[target]
    y_test = test_data[target]
    test_hours = test_data["hour"]

    return X_train, X_test, y_train, y_test, test_hours


def evaluate_fitted_regressor(
    model_name: str,
    fitted_model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    test_hours: pd.Series,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate a fitted regression pipeline and create test predictions."""
    y_pred_train = fitted_model.predict(X_train)
    y_pred_test = fitted_model.predict(X_test)

    metrics = {
        "train_mae": float(
            mean_absolute_error(y_train, y_pred_train)
        ),
        "test_mae": float(
            mean_absolute_error(y_test, y_pred_test)
        ),
        "train_rmse": float(
            np.sqrt(mean_squared_error(y_train, y_pred_train))
        ),
        "test_rmse": float(
            np.sqrt(mean_squared_error(y_test, y_pred_test))
        ),
        "train_r2": float(
            r2_score(y_train, y_pred_train)
        ),
        "test_r2": float(
            r2_score(y_test, y_pred_test)
        ),
        "negative_predictions": int(
            (y_pred_test < 0).sum()
        ),
    }

    predictions = pd.DataFrame(
        {
            "hour": test_hours.reset_index(drop=True),
            "actual_total_count": y_test.reset_index(drop=True),
            "predicted_total_count": y_pred_test,
        }
    )

    predictions["residual"] = (
        predictions["actual_total_count"]
        - predictions["predicted_total_count"]
    )

    predictions["model"] = model_name

    return metrics, predictions



def validate_comparable_prediction_tables(prediction_tables: dict[str, pd.DataFrame]) -> None:
    """Ensure models were evaluated on the same test observations."""
    if not prediction_tables:
        raise ValueError("No prediction tables were provided for comparison.")

    for model_name, predictions in prediction_tables.items():
        missing_columns = [
            column
            for column in REQUIRED_PREDICTION_COLUMNS
            if column not in predictions.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Prediction table for '{model_name}' is missing columns: "
                f"{missing_columns}"
            )

    reference_name, reference_predictions = next(
        iter(prediction_tables.items())
    )

    reference_test_rows = (
        reference_predictions[["hour", "actual_total_count"]]
        .sort_values("hour")
        .reset_index(drop=True)
    )

    for model_name, predictions in prediction_tables.items():
        current_test_rows = (
            predictions[["hour", "actual_total_count"]]
            .sort_values("hour")
            .reset_index(drop=True)
        )

        if not current_test_rows.equals(reference_test_rows):
            raise ValueError(
                f"Predictions from '{model_name}' cannot be compared with "
                f"'{reference_name}' because they do not use the same "
                "test observations."
            )


def summarize_test_predictions(model_name: str, predictions: pd.DataFrame) -> dict[str, str | float | int]:
    """Calculate regression metrics from test predictions."""
    y_true = predictions["actual_total_count"]
    y_pred = predictions["predicted_total_count"]

    return {
        "model": model_name,
        "test_mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "test_rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "test_r2": float(
            r2_score(y_true, y_pred)
        ),
        "negative_predictions": int(
            (y_pred < 0).sum()
        ),
    }


def build_model_comparison_table(prediction_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a ranking table for comparable model predictions."""
    validate_comparable_prediction_tables(prediction_tables)

    comparison_results = [
        summarize_test_predictions(model_name, predictions)
        for model_name, predictions in prediction_tables.items()
    ]

    return pd.DataFrame(comparison_results).sort_values("test_rmse").reset_index(drop=True)