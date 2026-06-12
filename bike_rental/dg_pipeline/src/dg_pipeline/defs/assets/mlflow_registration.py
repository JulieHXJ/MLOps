import mlflow
import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset
from mlflow.tracking import MlflowClient

from dg_pipeline.utils.mlflow_log import setup_mlflow


@asset(group_name="mlflow_registry")
def registered_production_model(context: AssetExecutionContext, engineered_model_comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically register the best engineered model as the production model.

    The best model is selected from engineered_model_comparison based on
    the lowest test_rmse.
    """
    registered_model_name = "BikeRentalDemandModel"
    production_alias = "production"

    setup_mlflow()
    required_columns = {
        "model",
        "test_mae",
        "test_rmse",
        "test_r2",
        "run_id",
    }
    
    missing_columns = required_columns - set(engineered_model_comparison.columns)
    if missing_columns:
        raise ValueError(
            f"engineered_model_comparison is missing columns: {missing_columns}"
        )

    comparison = engineered_model_comparison.copy()

    comparison["test_rmse"] = pd.to_numeric(
        comparison["test_rmse"],
        errors="coerce",
    )

    comparison = comparison.dropna(subset=["test_rmse", "run_id"])

    if comparison.empty:
        raise ValueError(
            "No valid model found in engineered_model_comparison."
        )

    best_model = comparison.sort_values("test_rmse", ascending=True).iloc[0]
    best_model_name = str(best_model["model"])
    best_run_id = str(best_model["run_id"])
    best_rmse = float(best_model["test_rmse"])
    best_mae = float(best_model["test_mae"])
    best_r2 = float(best_model["test_r2"])

    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    client = MlflowClient()
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=production_alias,
        version=model_version.version,
    )

    context.add_output_metadata(
        {
            "registered_model_name": registered_model_name,
            "production_alias": production_alias,
            "selected_model": best_model_name,
            "selected_run_id": best_run_id,
            "selected_test_rmse": best_rmse,
            "selected_test_mae": best_mae,
            "selected_test_r2": best_r2,
            "model_version": model_version.version,
            "model_uri": model_uri,
            "selection_logic": "lowest test_rmse from engineered_model_comparison",
            "selected_model_summary": MetadataValue.md(
                pd.DataFrame([best_model]).round(3).to_markdown(index=False)
            ),
        }
    )

    return pd.DataFrame(
        [
            {
                "registered_model_name": registered_model_name,
                "alias": production_alias,
                "selected_model": best_model_name,
                "run_id": best_run_id,
                "test_mae": best_mae,
                "test_rmse": best_rmse,
                "test_r2":best_r2,
                "model_version": model_version.version,
                "model_uri": model_uri,
            }
        ]
    )