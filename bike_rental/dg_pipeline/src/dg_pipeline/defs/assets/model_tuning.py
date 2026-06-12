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
    build_tuned_xgboost_pipeline,
)

from dg_pipeline.utils.mlflow_log import log_model_training_run

# @multi_asset(
#     group_name="model_tuning",
#     outs={
#         "xgboost_predictions": AssetOut(),
#     },
# )
# def tuned_model_predictions(
#     context: AssetExecutionContext,
#     engineered_X_train: pd.DataFrame,
#     engineered_X_test: pd.DataFrame,
#     engineered_y_train,
#     engineered_y_test,
#     engineered_test_hours,
# ):
#     """
#     Train XGBoost and Gradient Boost model candidates on the engineered feature set.
#     """

#     feature_config = FEATURE_SETS["engineered"]

#     y_train_series = to_series(engineered_y_train)
#     y_test_series = to_series(engineered_y_test)
#     test_hours_series = to_series(engineered_test_hours)

#     model_builders = {
#         "tuned_gradient_boosting_predictions": (
#             "Tuned Gradient Boosting",
#             build_tuned_gradient_boosting_pipeline,
#         ),
#         "xgboost_predictions": (
#             "XGBoost",
#             build_xgboost_pipeline,
#         ),
#         "tuned_xgboost_predictions": (
#             "Tuned XGBoost",
#             build_tuned_xgboost_pipeline,
#         ),
#     }

#     outputs = {}

#     for output_name, (model_name, pipeline_builder) in model_builders.items():
#         pipeline = pipeline_builder(
#             numeric_features=feature_config["numeric_features"],
#             categorical_features=feature_config["categorical_features"],
#         )

#         pipeline.fit(engineered_X_train, y_train_series)

#         evaluation_metrics, predictions = evaluate_regressor(
#             model_name=model_name,
#             fitted_model=pipeline,
#             X_train=engineered_X_train,
#             X_test=engineered_X_test,
#             y_train=y_train_series,
#             y_test=y_test_series,
#             test_hours=test_hours_series,
#         )

#         context.add_output_metadata(
#             {
#                 "model": model_name,
#                 "feature_set": "engineered",
#                 "feature_description": feature_config["description"],
#                 "split_strategy": "chronological 80/20 split",
#                 "target": feature_config["target"],
#                 "feature_count_before_encoding": len(
#                     feature_config["features"]
#                 ),
#                 "numeric_feature_count": len(
#                     feature_config["numeric_features"]
#                 ),
#                 "categorical_feature_count": len(
#                     feature_config["categorical_features"]
#                 ),
#                 "train_rows": len(engineered_X_train),
#                 "test_rows": len(engineered_X_test),
#                 **evaluation_metrics,
#                 "prediction_preview": MetadataValue.md(
#                     predictions.head().to_markdown(index=False)
#                 ),
#             },
#             output_name=output_name,
#         )

#         outputs[output_name] = predictions

#     return (
#         outputs["tuned_gradient_boosting_predictions"],
#         outputs["xgboost_predictions"],
#         outputs["tuned_xgboost_predictions"],
#     )


# @asset(group_name="model_evaluation")
# def tuned_model_comparison(
#     context: AssetExecutionContext,
#     tuned_gradient_boosting_predictions: pd.DataFrame,
#     xgboost_predictions: pd.DataFrame,
#     tuned_xgboost_predictions: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Compare tuned model candidates on engineered features.
#     """

#     comparison = build_model_comparison_table(
#         {
#             "Tuned Gradient Boosting": tuned_gradient_boosting_predictions,
#             "XGBoost": xgboost_predictions,
#             "Tuned XGBoost": tuned_xgboost_predictions,
#         }
#     )

#     best_model = comparison.iloc[0]["model"]
#     best_rmse = comparison.iloc[0]["test_rmse"]

#     context.add_output_metadata(
#         {
#             "feature_set": "engineered",
#             "experiment_stage": "model_tuning",
#             "compared_model_count": len(comparison),
#             "best_model": best_model,
#             "best_test_rmse": float(best_rmse),
#             "comparison_preview": MetadataValue.md(
#                 comparison.round(3).to_markdown(index=False)
#             ),
#         }
#     )

#     return comparison