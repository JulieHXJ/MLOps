import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset

# features
from dg_pipeline.modeling.baseline import (
    BASELINE_FEATURES,
    BASELINE_TARGET,
    build_linear_regression_baseline,
)

from dg_pipeline.modeling.evaluation import (
    build_model_comparison_table,
    evaluate_fitted_regressor,
    prepare_train_test_split,
)

# models
from dg_pipeline.modeling.model_candidates import (
    build_random_forest_candidate,
)


from dg_pipeline.modeling.engineered import (
    ENGINEERED_FEATURES,
    ENGINEERED_TARGET,
    build_engineered_gradient_boosting,
    build_engineered_linear_regression,
    build_engineered_random_forest,
)

@asset(group_name="model_training")
def baseline_linear_regression_predictions(
    context: AssetExecutionContext,
    model_ready_data: pd.DataFrame,
) -> pd.DataFrame:
    """Train the Linear Regression as reference and return test-set predictions."""
    X_train, X_test, y_train, y_test, test_hours = (
        prepare_train_test_split(
            model_ready_data=model_ready_data,
            features=BASELINE_FEATURES,
            target=BASELINE_TARGET,
        )
    )

    # train model
    baseline_pipeline = build_linear_regression_baseline()
    baseline_pipeline.fit(X_train, y_train)

    evaluation_metrics, predictions = evaluate_fitted_regressor(
        model_name="Linear Regression Baseline",
        fitted_model=baseline_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_hours=test_hours,
    )

    context.add_output_metadata(
        {
            "model": "LinearRegression",
            "model_role": "baseline reference",
            "feature_set": "V3: time + numeric weather + conditions",
            "split_strategy": "chronological 80/20 split",
            "target": BASELINE_TARGET,
            "feature_count_before_encoding": len(BASELINE_FEATURES),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            **evaluation_metrics,
            "prediction_preview": MetadataValue.md(
                predictions.head().to_markdown()
            ),
        }
    )

    return predictions


@asset(group_name="model_training")
def engineered_linear_regression_predictions(
    context: AssetExecutionContext,
    engineered_model_data: pd.DataFrame,
) -> pd.DataFrame:
    """Train Linear Regression with engineered features and return test predictions."""
    X_train, X_test, y_train, y_test, test_hours = (
        prepare_train_test_split(
            model_ready_data=engineered_model_data,
            features=ENGINEERED_FEATURES,
            target=ENGINEERED_TARGET,
        )
    )

    engineered_pipeline = build_engineered_linear_regression()
    engineered_pipeline.fit(X_train, y_train)

    evaluation_metrics, predictions = evaluate_fitted_regressor(
        model_name="Linear Regression Engineered Features",
        fitted_model=engineered_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_hours=test_hours,
    )

    context.add_output_metadata(
        {
            "model": "LinearRegression",
            "model_role": "engineered feature baseline",
            "feature_set": "engineered calendar + cyclical + historical features",
            "split_strategy": "chronological 80/20 split",
            "target": ENGINEERED_TARGET,
            "feature_count_before_encoding": len(ENGINEERED_FEATURES),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            **evaluation_metrics,
            "prediction_preview": MetadataValue.md(
                predictions.head().to_markdown()
            ),
        }
    )

    return predictions


@asset(group_name="model_training")
def baseline_random_forest_predictions(
    context: AssetExecutionContext,
    model_ready_data: pd.DataFrame,
) -> pd.DataFrame:
    """Train the selected Random Forest candidate and return test predictions."""
    X_train, X_test, y_train, y_test, test_hours = (
        prepare_train_test_split(
            model_ready_data=model_ready_data,
            features=BASELINE_FEATURES,
            target=BASELINE_TARGET,
        )
    )

    random_forest_pipeline = build_random_forest_candidate()
    random_forest_pipeline.fit(X_train, y_train)

    evaluation_metrics, predictions = evaluate_fitted_regressor(
        model_name="Random Forest V3 Candidate",
        fitted_model=random_forest_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_hours=test_hours,
    )

    context.add_output_metadata(
        {
            "model": "RandomForestRegressor",
            "model_role": "best candidate on V3 features",
            "feature_set": "V3: time + numeric weather + conditions",
            "split_strategy": "chronological 80/20 split",
            "target": BASELINE_TARGET,
            "feature_count_before_encoding": len(BASELINE_FEATURES),
            "n_estimators": 200,
            "max_depth": "None",
            "random_state": 42,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            **evaluation_metrics,
            "prediction_preview": MetadataValue.md(
                predictions.head().to_markdown()
            ),
        }
    )

    return predictions



@asset(group_name="model_training")
def engineered_random_forest_predictions(
    context: AssetExecutionContext,
    engineered_model_data: pd.DataFrame,
) -> pd.DataFrame:
    """Train Random Forest with engineered features and return test predictions."""
    X_train, X_test, y_train, y_test, test_hours = (
        prepare_train_test_split(
            model_ready_data=engineered_model_data,
            features=ENGINEERED_FEATURES,
            target=ENGINEERED_TARGET,
        )
    )

    engineered_pipeline = build_engineered_random_forest()
    engineered_pipeline.fit(X_train, y_train)

    evaluation_metrics, predictions = evaluate_fitted_regressor(
        model_name="Random Forest Engineered Features",
        fitted_model=engineered_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_hours=test_hours,
    )

    context.add_output_metadata(
        {
            "model": "RandomForestRegressor",
            "model_role": "engineered feature candidate",
            "feature_set": "engineered calendar + cyclical + historical features",
            "split_strategy": "chronological 80/20 split",
            "target": ENGINEERED_TARGET,
            "feature_count_before_encoding": len(ENGINEERED_FEATURES),
            "n_estimators": 200,
            "max_depth": "None",
            "random_state": 42,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            **evaluation_metrics,
            "prediction_preview": MetadataValue.md(
                predictions.head().to_markdown()
            ),
        }
    )

    return predictions



@asset(group_name="model_training")
def engineered_gradient_boosting_predictions(
    context: AssetExecutionContext,
    engineered_model_data: pd.DataFrame,
) -> pd.DataFrame:
    """Train Gradient Boosting with engineered features and return test predictions."""
    X_train, X_test, y_train, y_test, test_hours = (
        prepare_train_test_split(
            model_ready_data=engineered_model_data,
            features=ENGINEERED_FEATURES,
            target=ENGINEERED_TARGET,
        )
    )

    engineered_pipeline = build_engineered_gradient_boosting()
    engineered_pipeline.fit(X_train, y_train)

    evaluation_metrics, predictions = evaluate_fitted_regressor(
        model_name="Gradient Boosting Engineered Features",
        fitted_model=engineered_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_hours=test_hours,
    )

    context.add_output_metadata(
        {
            "model": "GradientBoostingRegressor",
            "model_role": "engineered feature candidate",
            "feature_set": "engineered calendar + cyclical + historical features",
            "split_strategy": "chronological 80/20 split",
            "target": ENGINEERED_TARGET,
            "feature_count_before_encoding": len(ENGINEERED_FEATURES),
            "random_state": 42,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            **evaluation_metrics,
            "prediction_preview": MetadataValue.md(
                predictions.head().to_markdown()
            ),
        }
    )

    return predictions


@asset(group_name="model_evaluation")
def baseline_model_comparison(
    context: AssetExecutionContext,
    baseline_linear_regression_predictions: pd.DataFrame,
    baseline_random_forest_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Linear Regression and Random Forest predictions on the same test period."""
    comparison = build_model_comparison_table(
        {
            "Linear Regression Baseline": baseline_linear_regression_predictions,
            "Random Forest Candidate": baseline_linear_regression_predictions,
        }
    )

    baseline_rmse = comparison.loc[
        comparison["model"] == "Linear Regression Baseline",
        "test_rmse",
    ].iloc[0]

    best_model = comparison.iloc[0]["model"]
    best_rmse = comparison.iloc[0]["test_rmse"]

    rmse_reduction_vs_baseline_pct = (
        (baseline_rmse - best_rmse)
        / baseline_rmse
        * 100
    )

    context.add_output_metadata(
        {
            "feature_set": "V3: time + numeric weather + conditions",
            "split_strategy": "chronological 80/20 split",
            "compared_model_count": len(comparison),
            "best_model": best_model,
            "rmse_reduction_vs_baseline_pct": float(
                rmse_reduction_vs_baseline_pct
            ),
            "comparison_preview": MetadataValue.md(
                comparison.round(3).to_markdown(index=False)
            ),
        }
    )

    return comparison


@asset(group_name="model_evaluation")
def engineered_model_comparison(
    context: AssetExecutionContext,
    engineered_linear_regression_predictions: pd.DataFrame,
    engineered_random_forest_predictions: pd.DataFrame,
    engineered_gradient_boosting_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare models trained with engineered features on the same test period."""
    comparison = build_model_comparison_table(
        {
            "Linear Regression Engineered Features": engineered_linear_regression_predictions,
            "Random Forest Engineered Features": engineered_random_forest_predictions,
            "Gradient Boosting Engineered Features": engineered_gradient_boosting_predictions,
        }
    )

    best_model = comparison.iloc[0]["model"]
    best_rmse = comparison.iloc[0]["test_rmse"]

    output_path = "../data/processed/engineered_model_comparison.csv"
    comparison.to_csv(output_path, index=False)

    context.add_output_metadata(
        {
            "output_path": MetadataValue.path(output_path),
            "feature_set": "engineered calendar + cyclical + historical features",
            "split_strategy": "chronological 80/20 split",
            "compared_model_count": len(comparison),
            "best_model": best_model,
            "best_test_rmse": float(best_rmse),
            "comparison_preview": MetadataValue.md(
                comparison.round(3).to_markdown(index=False)
            ),
        }
    )

    return comparison