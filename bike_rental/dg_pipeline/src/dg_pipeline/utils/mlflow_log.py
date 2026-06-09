import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd


def setup_mlflow() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("bike_rental_model_training")

def log_dataset_preview(
    dataset: pd.DataFrame,
    artifact_path: str = "dataset",
) -> None:
    """
    Log a small preview of the dataset as MLflow artifacts.
    This is useful for checking which feature table was used in a run.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        head_path = tmpdir_path / "dataset_head.csv"
        columns_path = tmpdir_path / "dataset_columns.txt"

        dataset.head().to_csv(head_path, index=False)

        with open(columns_path, "w", encoding="utf-8") as file:
            for column in dataset.columns:
                file.write(f"{column}\n")

        mlflow.log_artifact(str(head_path), artifact_path=artifact_path)
        mlflow.log_artifact(str(columns_path), artifact_path=artifact_path)


def log_model_training_run(
    model_name: str,
    feature_set: str,
    model_stage: str,
    pipeline,
    metrics: dict,
    train_rows: int,
    test_rows: int,
    feature_count_before_encoding: int,
    dataset: pd.DataFrame | None = None,
) -> str:
    """
    Log one model training run to MLflow.

    Returns
    -------
    str
        MLflow run id.
    """
    setup_mlflow()

    final_estimator = pipeline.steps[-1][1]

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("model_stage", model_stage)
        mlflow.log_param("model_type", type(final_estimator).__name__)

        mlflow.log_param("train_rows", train_rows)
        mlflow.log_param("test_rows", test_rows)
        mlflow.log_param(
            "feature_count_before_encoding",
            feature_count_before_encoding,
        )

        mlflow.log_params(final_estimator.get_params())

        mlflow.log_metrics(metrics)

        if dataset is not None:
            mlflow.log_param("dataset_rows", dataset.shape[0])
            mlflow.log_param("dataset_columns", dataset.shape[1])
            log_dataset_preview(dataset)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
        )

        return run.info.run_id