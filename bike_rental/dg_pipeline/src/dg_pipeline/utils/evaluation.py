from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REQUIRED_PREDICTION_COLUMNS = [
    "hour",
    "actual_total_count",
    "predicted_total_count",
]


def evaluate_regressor(
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

    train_log_metrics = calculate_log_error_metrics(
        y_train,
        y_pred_train,
    )

    test_log_metrics = calculate_log_error_metrics(
        y_test,
        y_pred_test,
    )

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
        "train_log_mae": train_log_metrics["log_mae"],
        "test_log_mae": test_log_metrics["log_mae"],
        "train_rmsle": train_log_metrics["rmsle"],
        "test_rmsle": test_log_metrics["rmsle"],

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



def calculate_log_error_metrics(
    y_true,
    y_pred,
) -> dict[str, float]:
    """
    Calculate log-scale error metrics.

    Negative predictions are clipped to 0 before log1p, because rental count
    cannot be negative.
    """

    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(np.clip(y_pred, a_min=0, a_max=None))

    log_error = y_true_log - y_pred_log

    return {
        "log_mae": float(np.mean(np.abs(log_error))),
        "rmsle": float(np.sqrt(np.mean(log_error ** 2))),
    }