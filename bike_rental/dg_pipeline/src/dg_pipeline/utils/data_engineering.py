import pandas as pd


HOURLY_INFORMATION_COLUMNS = [
    "date",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "conditions",
    "temperature_c",
    "humidity",
    "windspeed_kmh",
    "is_holiday",
    "is_workday",
]


AGGREGATION_RULES = {
    "direct_count": "sum",
    "registered_count": "sum",
    "date": "first",
    "hour_of_day": "first",
    "day_of_week": "first",
    "month": "first",
    "is_weekend": "first",
    "conditions": "first",
    "temperature_c": "first",
    "humidity": "first",
    "windspeed_kmh": "first",
    "is_holiday": "first",
    "is_workday": "first",
}


MODEL_READY_COLUMNS = [
    "hour",
    "direct_count",
    "registered_count",
    "total_count",
    "date",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "conditions",
    "temperature_c",
    "humidity",
    "windspeed_kmh",
    "is_holiday",
    "is_workday",
]

def validate_hourly_feature_consistency(data: pd.DataFrame) -> None:
    """Ensure hourly contextual features are identical across locations."""
    inconsistent_columns = []

    for column in HOURLY_INFORMATION_COLUMNS:
        has_different_values_within_hour = (
            data
            .groupby("hour")[column]
            .nunique(dropna=False)
            .gt(1)
            .any()
        )

        if has_different_values_within_hour:
            inconsistent_columns.append(column)

    if inconsistent_columns:
        raise ValueError(
            "These columns are not identical across locations "
            f"within the same hour: {inconsistent_columns}"
        )
    

def aggregate_city_hourly_demand(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate location-level rental demand into one row per hour."""
    hourly_data = (
        data
        .groupby("hour", as_index=False)
        .agg(AGGREGATION_RULES)
        .sort_values("hour")
        .reset_index(drop=True)
    )

    hourly_data["total_count"] = (
        hourly_data["direct_count"]
        + hourly_data["registered_count"]
    )

    return hourly_data[MODEL_READY_COLUMNS]

def validate_data(hourly_data: pd.DataFrame) -> dict[str, int]:
    """Validate aggregated hourly data and report remaining missing values."""
    duplicate_hours = int(hourly_data["hour"].duplicated().sum())

    if duplicate_hours > 0:
        raise ValueError(
            f"Hourly aggregation failed: {duplicate_hours} duplicate hours remain."
        )

    missing_by_column = hourly_data.isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0]

    if not missing_by_column.empty:
        raise ValueError(
            "Missing values remain in aggregated hourly data: "
            f"{missing_by_column.to_dict()}"
        )

    return {
        "remaining_duplicate_hours": duplicate_hours,
        "remaining_missing_values": int(hourly_data.isna().sum().sum()),
    }