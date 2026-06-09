import mlflow
import pandas as pd
from dagster import AssetExecutionContext, asset
from mlflow.tracking import MlflowClient

from dg_pipeline.utils.mlflow_log import setup_mlflow

MLFLOW_EXPERIMENT_NAME = "bike_rental_model_training"

@asset(group_name="mlfow_registry")
def registered_production_model(context: AssetExecutionContext,) -> pd.DataFrame:
    """
    Register the selected XGBoost engineered model as the production model.

    The model is loaded from the latest successful MLflow run named
    'XGBoost Engineered'.
    """

    registered_model_name = "BikeRentalDemandModel"
    selected_run_name = "XGBoost Engineered"

    setup_mlflow()
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(
            f"Experiment '{MLFLOW_EXPERIMENT_NAME}' was not found in MLflow."
        )


    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name = '{selected_run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(
            f"No MLflow run found with run name '{selected_run_name}'."
        )

    selected_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{selected_run_id}/model"

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

    client.set_registered_model_alias(
        name=registered_model_name,
        alias="production",
        version=model_version.version,
    )

    context.add_output_metadata(
        {
            "registered_model_name": registered_model_name,
            "production_alias": "production",
            "selected_model": selected_run_name,
            "selected_run_id": selected_run_id,
            "model_version": model_version.version,
            "model_uri": model_uri,
        }
    )

    return pd.DataFrame( 
        [
            {
                "registered_model_name": registered_model_name,
                "alias": "production",
                "selected_model": selected_run_name,
                "run_id": selected_run_id,
                "model_version": model_version.version,
                "model_uri": model_uri,
            }
        ]
    )