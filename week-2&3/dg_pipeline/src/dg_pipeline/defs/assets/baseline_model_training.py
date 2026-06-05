import pandas as pd
from dagster import (
    AssetExecutionContext,
    AssetOut,
    MetadataValue,
    asset,
    multi_asset,
)

from dg_pipeline.modeling.feature_config import FEATURE_SETS

from dg_pipeline.modeling.evaluation import (
    build_model_comparison_table,
    evaluate_regressor,
)


from dg_pipeline.modeling.model_pipeline import (
    build_gradient_boosting_pipeline,
    build_linear_regression_pipeline,
    build_random_forest_pipeline,
    create_train_test_split
)


@multi_asset(
    group_name="model_training",
    outs={
        "baseline_X_train": AssetOut(),
        "baseline_X_test": AssetOut(),
        "baseline_y_train": AssetOut(),
        "baseline_y_test": AssetOut(),
        "baseline_test_hours": AssetOut(),
    },
)
def baseline_train_test_data(
    context: AssetExecutionContext,
    enriched_rental_data: pd.DataFrame,
):
    """Create chronological train/test data for the baseline feature set."""

    feature_config = FEATURE_SETS["baseline"]

    (
        X_train,
        X_test,
        y_train,
        y_test,
        test_hours,
        train_data,
        test_data,
    ) = create_train_test_split(
        data=enriched_rental_data,
        feature_config=feature_config,
    )

    context.add_output_metadata(
        {
            "feature_set": "baseline",
            "split_strategy": "chronological 80/20 split",
            "target": feature_config["target"],
            "feature_count": len(feature_config["features"]),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_start": str(train_data["hour"].min()),
            "train_end": str(train_data["hour"].max()),
            "test_start": str(test_data["hour"].min()),
            "test_end": str(test_data["hour"].max()),
            "feature_preview": MetadataValue.md(
                X_train.head().to_markdown()
            ),
        }, output_name="baseline_X_train",
    )

    return X_train, X_test, y_train, y_test, test_hours




@multi_asset(
    group_name="model_training",
    outs={
        "baseline_linear_regression_predictions": AssetOut(),
        "baseline_random_forest_predictions": AssetOut(),
        "baseline_gradient_boosting_predictions": AssetOut(),
    },
)
def baseline_model_predictions(
    context: AssetExecutionContext,
    baseline_X_train: pd.DataFrame,
    baseline_X_test: pd.DataFrame,
    baseline_y_train,
    baseline_y_test,
    baseline_test_hours,
):
    """
    Train all baseline model candidates on the same baseline train/test split
    and return their test-set predictions.
    """

    feature_config = FEATURE_SETS["baseline"]

    def to_series(data) -> pd.Series:
        """Convert a one-column DataFrame or Series to Series."""
        if isinstance(data, pd.Series):
            return data

        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError(
                    "Expected a one-column DataFrame when converting to Series, "
                    f"but got {data.shape[1]} columns."
                )
            return data.iloc[:, 0]

        raise TypeError(f"Expected Series or DataFrame, got {type(data)}.")
    
    baseline_y_train_series = to_series(baseline_y_train)
    baseline_y_test_series = to_series(baseline_y_test)
    baseline_test_hours_series = to_series(baseline_test_hours)

    model_builders = {
        "baseline_linear_regression_predictions": (
            "Linear Regression Baseline",
            build_linear_regression_pipeline,
        ),
        "baseline_random_forest_predictions": (
            "Random Forest Baseline",
            build_random_forest_pipeline,
        ),
        "baseline_gradient_boosting_predictions": (
            "Gradient Boosting Baseline",
            build_gradient_boosting_pipeline,
        ),
    }

    outputs = {}

    for output_name, (model_name, pipeline_builder) in model_builders.items():
        pipeline = pipeline_builder(
            numeric_features=feature_config["numeric_features"],
            categorical_features=feature_config["categorical_features"],
        )

        pipeline.fit(baseline_X_train, baseline_y_train_series)

        evaluation_metrics, predictions = evaluate_regressor(
            model_name=model_name,
            fitted_model=pipeline,
            X_train=baseline_X_train,
            X_test=baseline_X_test,
            y_train=baseline_y_train_series,
            y_test=baseline_y_test_series,
            test_hours=baseline_test_hours_series,
        )

        context.add_output_metadata(
            {
                "model": model_name,
                "feature_set": "baseline",
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
                "train_rows": len(baseline_X_train),
                "test_rows": len(baseline_X_test),
                **evaluation_metrics,
                "prediction_preview": MetadataValue.md(
                    predictions.head().to_markdown()
                ),
            },
            output_name=output_name,
        )

        outputs[output_name] = predictions

    return (
        outputs["baseline_linear_regression_predictions"],
        outputs["baseline_random_forest_predictions"],
        outputs["baseline_gradient_boosting_predictions"],
    )



@asset(group_name="model_evaluation")
def baseline_model_comparison(
    context: AssetExecutionContext,
    baseline_linear_regression_predictions: pd.DataFrame,
    baseline_random_forest_predictions: pd.DataFrame,
    baseline_gradient_boosting_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare baseline models trained on the same baseline train/test split.
    """

    comparison = build_model_comparison_table(
        {
            "Linear Regression Baseline": baseline_linear_regression_predictions,
            "Random Forest Baseline": baseline_random_forest_predictions,
            "Gradient Boosting Baseline": baseline_gradient_boosting_predictions,
        }
    )

    baseline_rmse = comparison.loc[
        comparison["model"] == "Linear Regression Baseline",
        "test_rmse",
    ].iloc[0]

    best_model = comparison.iloc[0]["model"]
    best_rmse = comparison.iloc[0]["test_rmse"]

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
            "rmse_reduction_vs_linear_pct": float(
                rmse_reduction_vs_linear_pct
            ),
            "comparison_preview": MetadataValue.md(
                comparison.round(3).to_markdown(index=False)
            ),
        }
    )

    return comparison

