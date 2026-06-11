import pandas as pd
from dagster import (
    AssetExecutionContext,
    AssetOut,
    MetadataValue,
    asset,
    multi_asset,
)

from dg_pipeline.utils.feature_config import FEATURE_SETS

from dg_pipeline.utils.evaluation import (
    build_model_comparison_table,
    evaluate_regressor,
)


from dg_pipeline.utils.model_pipeline import (
    build_gradient_boosting_pipeline,
    build_linear_regression_pipeline,
    build_random_forest_pipeline,
    build_xgboost_pipeline,
    train_test_split_by_strategy,
    to_series,
)

from dg_pipeline.utils.mlflow_log import log_model_training_run
from dg_pipeline.utils.lakefs_config import get_lakefs_config, build_lakefs_metadata


@multi_asset(
    group_name="engineered_features",
    outs={
        "X_train_engineered": AssetOut(),
        "X_test_engineered": AssetOut(),
        "y_train_engineered": AssetOut(),
        "y_test_engineered": AssetOut(),
        "test_hours_engineered": AssetOut(),
    },
)
def engineered_train_test_data(
    context: AssetExecutionContext,
    lakefs_engineered_model_data: pd.DataFrame,
):
    """Create chronological train/test data for the engineered feature set."""
    (
        X_train,
        X_test,
        y_train,
        y_test,
        test_hours,
        train_data,
        test_data,
    ) = train_test_split_by_strategy(
        data=lakefs_engineered_model_data,
        feature_config=FEATURE_SETS["engineered"],
        split_strategy="chronological"
    )

    context.add_output_metadata(
        {
            "feature_set": "engineered",
            "split_strategy": "chronological 80/20 split",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_start": str(train_data["hour"].min()),
            "train_end": str(train_data["hour"].max()),
            "test_start": str(test_data["hour"].min()),
            "test_end": str(test_data["hour"].max()),
            "feature_preview": MetadataValue.md(
                X_train.head().to_markdown()
            ),
        },
        output_name="X_train_engineered",
    )

    return X_train, X_test, y_train, y_test, test_hours


@multi_asset(
    group_name="engineered_models",
    outs={
        "linear_regression_predictions_engineered": AssetOut(),
        "random_forest_predictions_engineered": AssetOut(),
        "gradient_boosting_predictions_engineered": AssetOut(),
        "xgboost_predictions_engineered": AssetOut(),
    },
)
def engineered_model_predictions(
    context: AssetExecutionContext,
    X_train_engineered: pd.DataFrame,
    X_test_engineered: pd.DataFrame,
    y_train_engineered,
    y_test_engineered,
    test_hours_engineered,
):
    """
    Train all model candidates on the engineered feature set.
    """

    feature_config = FEATURE_SETS["engineered"]
    split_strategy = "chronological"
    

    y_train_series = to_series(y_train_engineered)
    y_test_series = to_series(y_test_engineered)
    test_hours_series = to_series(test_hours_engineered)

    model_builders = {
        "linear_regression_predictions_engineered": (
            "Linear Regression Engineered",
            build_linear_regression_pipeline,
        ),
        "random_forest_predictions_engineered": (
            "Random Forest Engineered",
            build_random_forest_pipeline,
        ),
        "gradient_boosting_predictions_engineered": (
            "Gradient Boosting Engineered",
            build_gradient_boosting_pipeline,
        ),
        "xgboost_predictions_engineered": (
            "XGBoost Engineered",
            build_xgboost_pipeline,
        ),
    }
    lakefs_config = get_lakefs_config()
    lakefs_metadata = build_lakefs_metadata(lakefs_config)


    outputs = {}

    for output_name, (model_name, pipeline_builder) in model_builders.items():
        pipeline = pipeline_builder(
            numeric_features=feature_config["numeric_features"],
            categorical_features=feature_config["categorical_features"],
        )

        pipeline.fit(X_train_engineered, y_train_series)

        evaluation_metrics, predictions = evaluate_regressor(
            model_name=model_name,
            fitted_model=pipeline,
            X_train=X_train_engineered,
            X_test=X_test_engineered,
            y_train=y_train_series,
            y_test=y_test_series,
            test_hours=test_hours_series,
        )

        run_id = log_model_training_run(
            model_name=model_name,
            feature_set="engineered",
            model_stage="candidate",
            pipeline=pipeline,
            metrics=evaluation_metrics,
            train_rows=len(X_train_engineered),
            test_rows=len(X_test_engineered),
            feature_count_before_encoding=len(feature_config["features"]),
            dataset=X_train_engineered,
            lakefs_metadata=lakefs_metadata,
        )

        # Store MLflow run_id as column
        predictions["run_id"] = run_id
        predictions["model_name"] = model_name
        predictions["feature_set"] = "engineered"
        predictions["split_strategy"] = split_strategy

        context.add_output_metadata(
            {
                "model": model_name,
                "mlflow_run_id": run_id,
                "feature_set": "engineered",
                "feature_description": feature_config["description"],
                "split_strategy": "chronological 80/20 split",
                "target": feature_config["target"],
                "feature_count_before_encoding": len(
                    feature_config["features"]
                ),
                "numeric_feature_count": len(
                    feature_config["numeric_features"]
                ),
                "categorical_feature_count": len(
                    feature_config["categorical_features"]
                ),
                "train_rows": len(X_train_engineered),
                "test_rows": len(X_test_engineered),
                **evaluation_metrics,
                "prediction_preview": MetadataValue.md(
                    predictions.head().to_markdown()
                ),
            },
            output_name=output_name,
        )

        outputs[output_name] = predictions

    return (
        outputs["linear_regression_predictions_engineered"],
        outputs["random_forest_predictions_engineered"],
        outputs["gradient_boosting_predictions_engineered"],
        outputs["xgboost_predictions_engineered"],
    )

@asset(group_name="model_evaluation")
def engineered_model_comparison(
    context: AssetExecutionContext,
    linear_regression_predictions_engineered: pd.DataFrame,
    random_forest_predictions_engineered: pd.DataFrame,
    gradient_boosting_predictions_engineered: pd.DataFrame,
    xgboost_predictions_engineered: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare models trained on engineered features.
    """

    comparison = build_model_comparison_table(
        {
            "Linear Regression Engineered": linear_regression_predictions_engineered,
            "Random Forest Engineered": random_forest_predictions_engineered,
            "Gradient Boosting Engineered": gradient_boosting_predictions_engineered,
            "XGBoost Engineered": xgboost_predictions_engineered,
        }
    )

    best_model = comparison.iloc[0]["model"]
    best_rmse = comparison.iloc[0]["test_rmse"]
    best_run_id = comparison.iloc[0]["run_id"] 




    context.add_output_metadata(
        {
            "feature_set": "engineered",
            "split_strategy": "chronological 80/20 split",
            "compared_model_count": len(comparison),
            "best_model": best_model,
            "best_test_rmse": float(best_rmse),
            "best_run_id": best_run_id,
            "comparison_preview": MetadataValue.md(
                comparison.round(3).to_markdown(index=False)
            ),
        }
    )

    return comparison