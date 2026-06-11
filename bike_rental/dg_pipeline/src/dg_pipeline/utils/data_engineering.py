import pandas as pd
import numpy as np


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



def get_season(month: int) -> str:
    """Convert month number into season label."""
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "autumn"



def create_calendar_features_from_timestamp(
    timestamp: pd.Timestamp,
    is_holiday: int,
) -> dict[str, int | float | str]:
    """
    Create calendar and cyclical features for one timestamp.

    This function is suitable for API inference.
    """

    timestamp = pd.to_datetime(timestamp)

    hour_of_day = timestamp.hour
    day_of_week = timestamp.dayofweek
    month = timestamp.month

    is_weekend = int(day_of_week >= 5)
    is_workday = int((day_of_week < 5) and (is_holiday == 0))

    rush_hour = int(hour_of_day in [7, 8, 9, 16, 17, 18])

    return {
        "is_weekend": is_weekend,
        "is_holiday": int(is_holiday),
        "is_workday": is_workday,
        "rush_hour": rush_hour,
        "hour_sin": float(np.sin(2 * np.pi * hour_of_day / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour_of_day / 24)),
        "weekday_sin": float(np.sin(2 * np.pi * day_of_week / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * day_of_week / 7)),
        "month_sin": float(np.sin(2 * np.pi * (month - 1) / 12)),
        "month_cos": float(np.cos(2 * np.pi * (month - 1) / 12)),
        "season": get_season(month),
    }


def create_historical_demand_features_from_timestamp(
    timestamp: pd.Timestamp,
    historical_demand: pd.DataFrame,
) -> dict[str, float]:
    """
    Create historical demand features from hour and total_count

    The logic mirrors the training feature engineering:
    - total_lag_24h = total_count shifted by 24 hours
    - total_lag_7d = total_count shifted by 7 days
    - total_24h_mean = previous 24 hours mean
    - total_7d_mean = previous 7 days mean
    - same_hour_weekday_4w_mean = previous 1/2/3/4 week same-hour average
    """

    timestamp = pd.to_datetime(timestamp)

    history = historical_demand.copy()
    history["hour"] = pd.to_datetime(history["hour"])
    history = history.sort_values("hour").reset_index(drop=True)

    lag_24h_time = timestamp - pd.Timedelta(hours=24)
    lag_7d_time = timestamp - pd.Timedelta(days=7)

    lag_1w_time = timestamp - pd.Timedelta(days=7)
    lag_2w_time = timestamp - pd.Timedelta(days=14)
    lag_3w_time = timestamp - pd.Timedelta(days=21)
    lag_4w_time = timestamp - pd.Timedelta(days=28)

    def get_total_count_at(hour: pd.Timestamp) -> float:
        row = history.loc[history["hour"] == hour, "total_count"]

        if row.empty:
            raise ValueError(
                f"Missing historical demand row for {hour}."
            )

        return float(row.iloc[0])

    past_24h = history[
        (history["hour"] >= timestamp - pd.Timedelta(hours=24))
        & (history["hour"] < timestamp)
    ]

    past_7d = history[
        (history["hour"] >= timestamp - pd.Timedelta(days=7))
        & (history["hour"] < timestamp)
    ]

    if past_24h.empty:
        raise ValueError(
            f"Missing past 24h demand window before {timestamp}."
        )

    if past_7d.empty:
        raise ValueError(
            f"Missing past 7d demand window before {timestamp}."
        )

    same_hour_weekday_values = [
        get_total_count_at(lag_1w_time),
        get_total_count_at(lag_2w_time),
        get_total_count_at(lag_3w_time),
        get_total_count_at(lag_4w_time),
    ]

    return {
        "total_lag_24h": get_total_count_at(lag_24h_time),
        "total_lag_7d": get_total_count_at(lag_7d_time),
        "total_24h_mean": float(past_24h["total_count"].mean()),
        "total_7d_mean": float(past_7d["total_count"].mean()),
        "same_hour_weekday_4w_mean": float(
            np.mean(same_hour_weekday_values)
        ),
    }