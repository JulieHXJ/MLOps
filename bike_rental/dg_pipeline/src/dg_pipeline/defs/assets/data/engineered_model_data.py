from pathlib import Path

import pandas as pd
import numpy as np
from dagster import AssetExecutionContext, MetadataValue, asset

from dg_pipeline.utils.data_engineering import (
    aggregate_city_hourly_demand,
    validate_hourly_feature_consistency,
    validate_data,
)


DATA_DIR = Path("../data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def fill_missing_weather_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing weather values"""
    data = data.copy()

    weather_numeric_columns = [
        "temperature_c",
        "humidity",
        "windspeed_kmh",
    ]

    existing_weather_numeric_columns = [
        col for col in weather_numeric_columns
        if col in data.columns
    ]

    data[existing_weather_numeric_columns] = (
        data[existing_weather_numeric_columns]
        .interpolate(method="linear")
        .ffill()
        .bfill()
    )

    if "conditions" in data.columns:
        data["conditions"] = (
            data["conditions"]
            .ffill()
            .bfill()
            .fillna("Unknown")
        )

    return data

@asset(group_name="model_data")
def aggregated_hourly_data(context: AssetExecutionContext, enriched_rental_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate location-level rental data into city-level hourly demand."""
    data = enriched_rental_data.copy()
    data["hour"] = pd.to_datetime(data["hour"])
    data["date"] = pd.to_datetime(data["date"])

    validate_hourly_feature_consistency(data)

    hourly_data = aggregate_city_hourly_demand(data)

    hourly_data = fill_missing_weather_columns(hourly_data)

    hourly_data["is_holiday"] = hourly_data["is_holiday"].fillna(0).astype(int)

    hourly_data["is_workday"] = (
        (hourly_data["day_of_week"] < 5)
        & (hourly_data["is_holiday"] == 0)
    ).astype(int)


    validation_metadata = validate_data(hourly_data)

    output_path = PROCESSED_DATA_DIR / "aggregated_hourly_rental_data.csv"
    hourly_data.to_csv(output_path, index=False)

    context.add_output_metadata(
        {
            "output_path": MetadataValue.path(str(output_path.resolve())),
            "input_location_level_rows": len(data),
            "final_hourly_rows": len(hourly_data),
            "final_column_count": len(hourly_data.columns),
            **validation_metadata,
            "total_direct_rentals": int(
                hourly_data["direct_count"].sum()
            ),
            "total_registered_rentals": int(
                hourly_data["registered_count"].sum()
            ),
            "total_rentals": int(
                hourly_data["total_count"].sum()
            ),
            "preview": MetadataValue.md(
                hourly_data.head().to_markdown()
            ),
        }
    )


    return hourly_data


def get_season(month: int) -> str:
    """Convert month number into season label."""
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "autumn"


def fill_missing_count(
    data: pd.DataFrame,
    count_columns: list[str],
) -> pd.DataFrame:
    """Fill missing count values using weekday-hour mean, hour mean, then global mean."""
    data = data.copy()

    for col in count_columns:
        weekday_hour_mean = (
            data.groupby(["day_of_week", "hour_of_day"])[col]
            .transform("mean")
        )
        data[col] = data[col].fillna(weekday_hour_mean)

    for col in count_columns:
        hour_mean = (
            data.groupby("hour_of_day")[col]
            .transform("mean")
        )
        data[col] = data[col].fillna(hour_mean)

    for col in count_columns:
        data[col] = data[col].fillna(data[col].mean())

    return data


def create_engineered_features(model_ready_data: pd.DataFrame) -> pd.DataFrame:
    """Create calendar, cyclical, and historical demand features."""
    data = model_ready_data.copy()

    data["hour"] = pd.to_datetime(data["hour"])
    data = data.sort_values("hour").reset_index(drop=True)

    # Reindex to complete hourly time series
    data = data.set_index("hour").sort_index()

    full_hour_index = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq="h",
    )

    data = data.reindex(full_hour_index)

    data.index.name = "hour"
    data = data.reset_index()

    # Recreate time-based columns after reindexing
    data["date"] = data["hour"].dt.date
    data["hour_of_day"] = data["hour"].dt.hour
    data["day_of_week"] = data["hour"].dt.dayofweek
    data["month"] = data["hour"].dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    # Track inserted rows
    data["was_missing_hour"] = data["total_count"].isna().astype(int)

    # Fill rental counts for inserted missing hours
    count_columns = [
        "direct_count",
        "registered_count",
        "total_count",
    ]

    data = fill_missing_count(
        data=data,
        count_columns=count_columns,
    )

    # Fill weather values for inserted missing hours
    data = fill_missing_weather_columns(data)

    # Recreate holiday/workday columns
    data["is_holiday"] = data["is_holiday"].fillna(0).astype(int)

    data["is_workday"] = (
        (data["day_of_week"] < 5)
        & (data["is_holiday"] == 0)
    ).astype(int)

    # Calendar-based features
    data["rush_hour"] = data["hour_of_day"].isin(
        [7, 8, 9, 16, 17, 18]
    ).astype(int)

    data["season"] = data["month"].apply(get_season)

    # Cyclical encoding
    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour_of_day"] / 24
    )
    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour_of_day"] / 24
    )

    data["weekday_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )
    data["weekday_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["month_sin"] = np.sin(
        2 * np.pi * (data["month"] - 1) / 12
    )
    data["month_cos"] = np.cos(
        2 * np.pi * (data["month"] - 1) / 12
    )

    # Historical lag features
    data["total_lag_24h"] = data["total_count"].shift(24)
    data["total_lag_7d"] = data["total_count"].shift(24 * 7)

    # Rolling historical features
    data["total_24h_mean"] = (
        data["total_count"]
        .shift(1)
        .rolling(window=24)
        .mean()
    )

    data["total_7d_mean"] = (
        data["total_count"]
        .shift(1)
        .rolling(window=24 * 7)
        .mean()
    )

    # Same hour and same weekday previous 4-week average
    data["total_count_lag_1w"] = data["total_count"].shift(24 * 7)
    data["total_count_lag_2w"] = data["total_count"].shift(24 * 7 * 2)
    data["total_count_lag_3w"] = data["total_count"].shift(24 * 7 * 3)
    data["total_count_lag_4w"] = data["total_count"].shift(24 * 7 * 4)

    lag_week_columns = [
        "total_count_lag_1w",
        "total_count_lag_2w",
        "total_count_lag_3w",
        "total_count_lag_4w",
    ]

    data["same_hour_weekday_4w_mean"] = (
        data[lag_week_columns].mean(axis=1)
    )

    return data


@asset(group_name="model_data")
def engineered_model_data(
    context: AssetExecutionContext,
    aggregated_hourly_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create feature-engineered model dataset after model_ready_data."""
    data = create_engineered_features(aggregated_hourly_data)

    target = "total_count"

    numeric_features = [
        "temperature_c",
        "humidity",
        "windspeed_kmh",

        "is_weekend",
        "is_holiday",
        "is_workday",
        "rush_hour",
        "was_missing_hour",

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
    ]

    categorical_features = [
        "conditions",
        "season",
    ]

    features = numeric_features + categorical_features

    engineered_data = data[
        ["hour", *features, target]
    ].copy()

    engineered_data = (
        engineered_data
        .dropna(subset=features + [target])
        .reset_index(drop=True)
    )

    output_path = PROCESSED_DATA_DIR / "engineered_model_data.csv"
    engineered_data.to_csv(output_path, index=False)

    context.add_output_metadata(
        {
            "output_path": MetadataValue.path(str(output_path.resolve())),
            "input_rows": len(aggregated_hourly_data),
            "engineered_rows": len(engineered_data),
            "feature_count": len(features),
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
            "target": target,
            "inserted_missing_hours": int(data["was_missing_hour"].sum()),
            "missing_values_after_cleaning": int(
                engineered_data.isna().sum().sum()
            ),
            "preview": MetadataValue.md(
                engineered_data.head().to_markdown(index=False)
            ),
        }
    )

    return engineered_data