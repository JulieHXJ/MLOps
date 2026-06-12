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
    train_test_split_by_strategy,
)

from dg_pipeline.utils.mlflow_log import log_model_training_run
from dg_pipeline.utils.lakefs_config import get_lakefs_config, build_lakefs_metadata


# @multi_asset(
#     group_name="baseline_features",
#     outs={
#         "X_train_baseline": AssetOut(),
#         "X_test_baseline": AssetOut(),
#         "y_train_baseline": AssetOut(),
#         "y_test_baseline": AssetOut(),
#         "test_hours_baseline": AssetOut(),
#     },
# )
# def baseline_train_test_data(
#     context: AssetExecutionContext,
#     aggregated_hourly_data: pd.DataFrame,
# ):
#     """Create chronological train/test data for the baseline feature set."""
#     (
#         X_train,
#         X_test,
#         y_train,
#         y_test,
#         test_hours,
#         train_data,
#         test_data,
#     ) = time_based_train_test_split(
#         data=aggregated_hourly_data,
#         feature_config=FEATURE_SETS["baseline"],
#     ) 

#     context.add_output_metadata(
#         {
#             "feature_set": "baseline",
#             "split_strategy": "chronological 80/20 split",
#             "train_rows": len(X_train),
#             "test_rows": len(X_test),
#             "train_start": str(train_data["hour"].min()),
#             "train_end": str(train_data["hour"].max()),
#             "test_start": str(test_data["hour"].min()),
#             "test_end": str(test_data["hour"].max()),
#             "feature_preview": MetadataValue.md(
#                 X_train.head().to_markdown()
#             ),
#         }, output_name="X_train_baseline",
#     )

#     return X_train, X_test, y_train, y_test, test_hours




@multi_asset(
    group_name="baseline_models",
    outs={
        "linear_regression_predictions_baseline": AssetOut(),
        "random_forest_predictions_baseline": AssetOut(),
        "gradient_boosting_predictions_baseline": AssetOut(),
    },
)
def baseline_model_predictions(
    context: AssetExecutionContext,
    aggregated_hourly_data: pd.DataFrame,
):
    """
    Train all baseline model candidates on the same baseline train/test split
    and return their test-set predictions.
    """

    feature_config = FEATURE_SETS["baseline"]
    split_strategy = "chronological"
    
    train_data_baseline, test_data_baseline = train_test_split_by_strategy(
        data=aggregated_hourly_data,
        split_strategy=split_strategy,
    )
    
    features = feature_config["features"]
    target = feature_config["target"]

    X_train_baseline = train_data_baseline[features]
    X_test_baseline = test_data_baseline[features]
    y_train_series = train_data_baseline[target]
    y_test_series = test_data_baseline[target]
    test_hours_series = test_data_baseline["hour"]



    model_builders = {
        "linear_regression_predictions_baseline": (
            "Linear Regression Baseline",
            build_linear_regression_pipeline,
        ),
        "random_forest_predictions_baseline": (
            "Random Forest Baseline",
            build_random_forest_pipeline,
        ),
        "gradient_boosting_predictions_baseline": (
            "Gradient Boosting Baseline",
            build_gradient_boosting_pipeline,
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

        pipeline.fit(X_train_baseline, y_train_series)

        evaluation_metrics, predictions = evaluate_regressor(
            model_name=model_name,
            fitted_model=pipeline,
            X_train=X_train_baseline,
            X_test=X_test_baseline,
            y_train=y_train_series,
            y_test=y_test_series,
            test_hours=test_hours_series,
        )

        # mlflow log
        run_id = log_model_training_run(
            model_name=model_name,
            feature_set="baseline",
            model_stage="candidate",
            pipeline=pipeline,
            metrics=evaluation_metrics,
            train_rows=len(X_train_baseline),
            test_rows=len(X_test_baseline),
            feature_count_before_encoding=len(feature_config["features"]),
            dataset=X_train_baseline,
            lakefs_metadata=lakefs_metadata,
        )

        predictions["run_id"] = run_id
        predictions["model_name"] = model_name
        predictions["feature_set"] = "baseline"
        predictions["split_strategy"] = split_strategy


        context.add_output_metadata(
            {
                "model": model_name,
            "mlflow_run_id": run_id,
            "feature_set": "baseline",
            "target": feature_config["target"],
            "feature_count_before_encoding": len(feature_config["features"]),
            "numeric_feature_count": len(feature_config["numeric_features"]),
            "categorical_feature_count": len(feature_config["categorical_features"]),
            "train_rows": len(X_train_baseline),
            "test_rows": len(X_test_baseline),
            **evaluation_metrics,
            "prediction_preview": MetadataValue.md(
                predictions.head().to_markdown()
                ),
            },
            output_name=output_name,
        )

        outputs[output_name] = predictions

    return (
        outputs["linear_regression_predictions_baseline"],
        outputs["random_forest_predictions_baseline"],
        outputs["gradient_boosting_predictions_baseline"],
    )



@asset(group_name="model_evaluation")
def baseline_model_comparison(
    context: AssetExecutionContext,
    linear_regression_predictions_baseline: pd.DataFrame,
    random_forest_predictions_baseline: pd.DataFrame,
    gradient_boosting_predictions_baseline: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare baseline models trained on the same baseline train/test split.
    """

    comparison = build_model_comparison_table(
        {
            "Linear Regression Baseline": linear_regression_predictions_baseline,
            "Random Forest Baseline": random_forest_predictions_baseline,
            "Gradient Boosting Baseline": gradient_boosting_predictions_baseline,
        }
    )

    baseline_rmse = comparison.loc[
        comparison["model"] == "Linear Regression Baseline",
        "test_rmse",
    ].iloc[0]

    best_model = comparison.iloc[0]["model"]
    best_rmse = comparison.iloc[0]["test_rmse"]
    best_run_id = comparison.iloc[0]["run_id"]

    rmse_reduction_vs_linear_pct = (
        (baseline_rmse - best_rmse)
        / baseline_rmse
        * 100
    )

    context.add_output_metadata(
        {
            "feature_set": "baseline",
            "split_strategy": "chronological 80/20 split",
            "compared_model_count": len(comparison),
            "best_model": best_model,
            "best_test_rmse": float(best_rmse),
            "best_run_id": best_run_id,
            "rmse_reduction_vs_linear_pct": float(
                rmse_reduction_vs_linear_pct
            ),
            "comparison_preview": MetadataValue.md(
                comparison.round(3).to_markdown(index=False)
            ),
        }
    )

    return comparison
