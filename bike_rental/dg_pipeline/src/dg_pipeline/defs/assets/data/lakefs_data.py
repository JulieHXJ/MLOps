import pandas as pd
import pandas as pd
from dagster import (
    AssetKey,
    AssetCheckResult,
    AssetExecutionContext,
    MetadataValue,
    asset,
    asset_check,
)

from dg_pipeline.utils.lakefs_config import get_lakefs_config
from dg_pipeline.utils.lakefs_io import build_lakefs_uri, read_csv_from_lakefs, write_csv_to_lakefs

from dagster import AssetExecutionContext, MetadataValue, asset
import pandas as pd


@asset(group_name="lakefs_data")
def upload_engineered_model_data_to_lakefs(
    context: AssetExecutionContext,
    engineered_model_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Upload engineered model data to LakeFS so downstream assets can use
    a versioned dataset source.
    """

    config = get_lakefs_config()
    lakefs_uri = build_lakefs_uri(config)

    write_csv_to_lakefs(
        data=engineered_model_data,
        config=config,
    )

    context.add_output_metadata(
        {
            "data_source": "dagster_engineered_model_data",
            "lakefs_uri": lakefs_uri,
            "lakefs_repo": config.repo,
            "lakefs_branch": config.branch,
            "lakefs_dataset_path": config.dataset_path,
            "lakefs_commit_id": config.commit_id or "not_set",
            "rows": len(engineered_model_data),
            "columns": len(engineered_model_data.columns),
            "preview": MetadataValue.md(
                engineered_model_data.head().to_markdown()
            ),
        }
    )

    return pd.DataFrame(
        [
            {
                "lakefs_uri": lakefs_uri,
                "lakefs_repo": config.repo,
                "lakefs_branch": config.branch,
                "lakefs_dataset_path": config.dataset_path,
                "lakefs_commit_id": config.commit_id or "not_set",
                "rows": len(engineered_model_data),
                "columns": len(engineered_model_data.columns),
            }
        ]
    )


@asset(group_name="lakefs_data", deps=[AssetKey("upload_engineered_model_data_to_lakefs")],)
def lakefs_engineered_model_data(
    context: AssetExecutionContext,
) -> pd.DataFrame:
    """
    Load the production model-ready engineered dataset from LakeFS.
    """

    config = get_lakefs_config()
    data = read_csv_from_lakefs(config)
    lakefs_uri = build_lakefs_uri(config)

    context.add_output_metadata(
        {
            "data_source": "lakefs",
            "lakefs_uri": lakefs_uri,
            "lakefs_repo": config.repo,
            "lakefs_branch": config.branch,
            "lakefs_dataset_path": config.dataset_path,
            "lakefs_commit_id": config.commit_id or "not_set",
            "rows": len(data),
            "columns": len(data.columns),
            "preview": MetadataValue.md(data.head().to_markdown()),
        }
    )

    return data


# add asset check
REQUIRED_ENGINEERED_COLUMNS = {
    "hour",
    "temperature_c",
    "humidity",
    "windspeed_kmh",
    "is_weekend",
    "is_holiday",
    "is_workday",
    "rush_hour",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "total_lag_24h",
    "total_lag_7d",
    "total_24h_mean",
    "total_7d_mean",
    "same_hour_weekday_4w_mean",
    "conditions",
    "season",
    "total_count",
}

@asset_check(asset=lakefs_engineered_model_data)
def lakefs_engineered_model_data_not_empty(
    lakefs_engineered_model_data: pd.DataFrame,
) -> AssetCheckResult:
    """Check that the LakeFS engineered model dataset is not empty."""

    row_count = len(lakefs_engineered_model_data)

    return AssetCheckResult(
        passed=bool(row_count > 0),
        metadata={
            "row_count": int(row_count),
        },
    )


@asset_check(asset=lakefs_engineered_model_data)
def lakefs_engineered_model_data_required_columns(
    lakefs_engineered_model_data: pd.DataFrame,
) -> AssetCheckResult:
    """Check that all required engineered model columns exist."""

    actual_columns = set(lakefs_engineered_model_data.columns)
    missing_columns = sorted(REQUIRED_ENGINEERED_COLUMNS - actual_columns)

    return AssetCheckResult(
        passed=bool(len(missing_columns) == 0),
        metadata={
            "required_column_count": int(len(REQUIRED_ENGINEERED_COLUMNS)),
            "actual_column_count": int(len(actual_columns)),
            "missing_columns": ", ".join(missing_columns)
            if missing_columns
            else "none",
        },
    )


@asset_check(asset=lakefs_engineered_model_data)
def lakefs_engineered_model_data_target_not_missing(
    lakefs_engineered_model_data: pd.DataFrame,
) -> AssetCheckResult:
    """Check that the target column has no missing values."""

    target_column = "total_count"
    missing_count = int(
        lakefs_engineered_model_data[target_column].isna().sum()
    )

    return AssetCheckResult(
        passed=bool(missing_count == 0),
        metadata={
            "target_column": target_column,
            "missing_target_count": missing_count,
        },
    )


@asset_check(asset=lakefs_engineered_model_data)
def lakefs_engineered_model_data_target_non_negative(
    lakefs_engineered_model_data: pd.DataFrame,
) -> AssetCheckResult:
    """Check that rental demand is never negative."""

    target_column = "total_count"
    negative_count = int(
        (lakefs_engineered_model_data[target_column] < 0).sum()
    )

    return AssetCheckResult(
        passed=bool(negative_count == 0),
        metadata={
            "target_column": target_column,
            "negative_target_count": negative_count,
            "min_target_value": float(
                lakefs_engineered_model_data[target_column].min()
            ),
        },
    )


@asset_check(asset=lakefs_engineered_model_data)
def lakefs_engineered_model_data_hour_parseable(
    lakefs_engineered_model_data: pd.DataFrame,
) -> AssetCheckResult:
    """Check that the hour column can be parsed as datetime."""

    parsed_hour = pd.to_datetime(
        lakefs_engineered_model_data["hour"],
        errors="coerce",
    )

    invalid_count = int(parsed_hour.isna().sum())

    return AssetCheckResult(
        passed=bool(invalid_count == 0),
        metadata={
            "invalid_hour_count": invalid_count,
            "min_hour": str(parsed_hour.min()),
            "max_hour": str(parsed_hour.max()),
        },
    )


