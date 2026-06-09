
# import pandas as pd
# from dagster import AssetExecutionContext, MetadataValue, asset

# from dg_pipeline.utils.evaluation import build_model_comparison_table


# @asset(group_name="model_evaluation")
# def final_model_evaluation(
#     context: AssetExecutionContext,
#     baseline_model_comparison: pd.DataFrame,
#     engineered_model_comparison: pd.DataFrame,
#     tuned_model_comparison: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Combine baseline, engineered, and tuned model comparison tables.
#     """

#     baseline = baseline_model_comparison.copy()
#     baseline["experiment_stage"] = "baseline_features"

#     engineered = engineered_model_comparison.copy()
#     engineered["experiment_stage"] = "engineered_features"

#     tuned = tuned_model_comparison.copy()
#     tuned["experiment_stage"] = "tuned_engineered_models"

#     final_comparison = pd.concat(
#         [
#             baseline,
#             engineered,
#             tuned,
#         ],
#         ignore_index=True,
#     )

#     final_comparison = (
#         final_comparison
#         .sort_values("test_rmse")
#         .reset_index(drop=True)
#     )

#     best_model = final_comparison.iloc[0]["model"]
#     best_rmse = final_comparison.iloc[0]["test_rmse"]

#     best_baseline_rmse = baseline["test_rmse"].min()
#     best_engineered_rmse = engineered["test_rmse"].min()

#     rmse_reduction_vs_baseline_pct = (
#         (best_baseline_rmse - best_rmse)
#         / best_baseline_rmse
#         * 100
#     )

#     rmse_reduction_vs_engineered_pct = (
#         (best_engineered_rmse - best_rmse)
#         / best_engineered_rmse
#         * 100
#     )

#     context.add_output_metadata(
#         {
#             "total_model_count": len(final_comparison),
#             "best_model": best_model,
#             "best_test_rmse": float(best_rmse),
#             "best_baseline_rmse": float(best_baseline_rmse),
#             "best_engineered_rmse": float(best_engineered_rmse),
#             "rmse_reduction_vs_baseline_pct": float(
#                 rmse_reduction_vs_baseline_pct
#             ),
#             "rmse_reduction_vs_engineered_pct": float(
#                 rmse_reduction_vs_engineered_pct
#             ),
#             "comparison_preview": MetadataValue.md(
#                 final_comparison.round(3).to_markdown(index=False)
#             ),
#         }
#     )

#     return final_comparison


# def classify_feature_set(model_name: str) -> str:
#     """Classify model by feature set / experiment stage."""
#     if "Baseline" in model_name:
#         return "baseline"

#     if "Tuned" in model_name or "XGBoost" in model_name:
#         return "tuned_engineered"

#     return "engineered"


# def classify_model_family(model_name: str) -> str:
#     """Classify model by algorithm family."""
#     if "Linear Regression" in model_name:
#         return "linear_regression"

#     if "Random Forest" in model_name:
#         return "random_forest"

#     if "XGBoost" in model_name:
#         return "xgboost"

#     if "Gradient Boosting" in model_name:
#         return "gradient_boosting"

#     return "unknown"