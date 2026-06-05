import numpy as np
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
        "engineered_X_train": AssetOut(),
        "engineered_X_test": AssetOut(),
        "engineered_y_train": AssetOut(),
        "engineered_y_test": AssetOut(),
        "engineered_test_hours": AssetOut(),
    },
)
def engineered_train_test_data(
    context: AssetExecutionContext,
    engineered_rental_data: pd.DataFrame,
):
    """
    Create chronological train/test data for the engineered feature set.
    """

    feature_config = FEATURE_SETS["engineered"]

    (
        X_train,
        X_test,
        y_train,
        y_test,
        test_hours,
        train_data,
        test_data,
    ) = create_train_test_split(
        data=engineered_rental_data,
        feature_config=feature_config,
    )

    context.add_output_metadata(
        {
            "feature_set": "engineered",
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
        }
    )

    return X_train, X_test, y_train, y_test, test_hours


