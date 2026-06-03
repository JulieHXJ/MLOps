import pandas as pd


HOURLY_INFORMATION_COLUMNS = [
    "date",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "conditions",
    "temperature_c",
    "perceived_temperature_c",
    "humidity",
    "windspeed_kmh",
    "holiday",
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
    "perceived_temperature_c": "first",
    "humidity": "first",
    "windspeed_kmh": "first",
    "holiday": "first",
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
    "perceived_temperature_c",
    "humidity",
    "windspeed_kmh",
    "holiday",
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

def validate_model_ready_data(hourly_data: pd.DataFrame) -> dict[str, int]:
    """Validate that the final model dataset has unique hours and no missing values."""
    duplicate_hours = int(
        hourly_data["hour"].duplicated().sum()
    )

    remaining_missing_values = int(
        hourly_data.isna().sum().sum()
    )

    if duplicate_hours > 0:
        raise ValueError(
            f"Hourly aggregation failed: {duplicate_hours} duplicate hours remain."
        )

    if remaining_missing_values > 0:
        raise ValueError(
            "Missing values remain in model-ready hourly data."
        )

    return {
        "remaining_duplicate_hours": duplicate_hours,
        "remaining_missing_values": remaining_missing_values,
    }