import pandas as pd
from pathlib import Path
from dagster import asset, AssetExecutionContext, MetadataValue


DATA_DIR = Path("../data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

@asset
def registered_rentals() -> pd.DataFrame:
    registered = pd.read_csv(DATA_DIR/"registered_bike_rentals.csv")
    registered["datetime"] = pd.to_datetime(registered["datetime"]) #convert datetime
    registered["hour"] = registered["datetime"].dt.floor("h") # create hour feature
    return registered

@asset
def direct_pickups() -> pd.DataFrame:
    direct = pd.read_csv(DATA_DIR/"direct_pickup_bike_rentals.csv")
    direct["datetime"] = pd.to_datetime(direct["datetime"])
    direct["hour"] = direct["datetime"].dt.floor("h")
    return direct

@asset
def weather_data() -> pd.DataFrame:
    weather = pd.read_csv(DATA_DIR/"weather.csv")
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    weather["hour"] = weather["datetime"].dt.floor("h")
    return weather

@asset
def holidays_data() -> pd.DataFrame:
    holidays = pd.read_csv(DATA_DIR/"holidays.csv")
    holidays["date"] = pd.to_datetime(holidays["date"]).dt.date
    return holidays

@asset
def all_rental_events(context: AssetExecutionContext, registered_rentals: pd.DataFrame, direct_pickups: pd.DataFrame) -> pd.DataFrame:
    registered_events = registered_rentals[["hour", "location_id"]].copy()
    registered_events["rental_type"] = "registered"

    direct_events = direct_pickups[["hour", "location_id"]].copy()
    direct_events["rental_type"] = "direct"

    all_events = pd.concat([registered_events, direct_events], ignore_index=True)
    return all_events


@asset
def hourly_location_rentals(context: AssetExecutionContext, all_rental_events: pd.DataFrame) -> pd.DataFrame:
    # group up
    hourly_location = (all_rental_events.groupby(["hour", "location_id", "rental_type"]).size().unstack(fill_value=0).reset_index())

    if "registered" not in hourly_location.columns:
        hourly_location["registered"] = 0

    if "direct" not in hourly_location.columns:
        hourly_location["direct"] = 0

    hourly_location = hourly_location.rename(
        columns={
            "registered": "registered_count",
            "direct": "direct_count",
        }
    )

    hourly_location["total_count"] = (
        hourly_location["registered_count"]
        + hourly_location["direct_count"]
    )

    hourly_location = (
        hourly_location
        .sort_values(["hour", "location_id"])
        .reset_index(drop=True)
    )

    # Create a complete hour-location grid.
    full_hours = pd.DataFrame(
        {
            "hour": pd.date_range(
                start=hourly_location["hour"].min(),
                end=hourly_location["hour"].max(),
                freq="h",
            )
        }
    )

    all_locations = pd.DataFrame(
        {
            "location_id": (
                all_rental_events["location_id"]
                .drop_duplicates()
                .sort_values()
            )
        }
    )

    full_table = full_hours.merge(all_locations, how="cross")

    hourly_location = full_table.merge(
        hourly_location,
        on=["hour", "location_id"],
        how="left",
    )

    # mark rows with no original rental and filled with 0.
    hourly_location["is_zero_filled_rental"] = (
        hourly_location["total_count"].isna().astype(int)
    )

    count_cols = ["registered_count", "direct_count", "total_count"]

    hourly_location[count_cols] = (
        hourly_location[count_cols]
        .fillna(0)
        .astype(int)
    )

    # add time-based features.
    hourly_location["date"] = hourly_location["hour"].dt.date
    hourly_location["hour_of_day"] = hourly_location["hour"].dt.hour
    hourly_location["day_of_week"] = hourly_location["hour"].dt.dayofweek
    hourly_location["month"] = hourly_location["hour"].dt.month
    hourly_location["is_weekend"] = (
        hourly_location["day_of_week"].isin([5, 6]).astype(int)
    )

    context.add_output_metadata(
        {
            "row_count": len(hourly_location),
            "expected_full_hour_location_rows": len(full_table),
            "location_count": int(all_locations["location_id"].nunique()),
            "registered_count_sum": int(
                hourly_location["registered_count"].sum()
            ),
            "direct_count_sum": int(
                hourly_location["direct_count"].sum()
            ),
            "total_count_sum": int(
                hourly_location["total_count"].sum()
            ),
            "zero_filled_rental_rows": int(
                hourly_location["is_zero_filled_rental"].sum()
            ),
            "preview": MetadataValue.md(
                hourly_location.head().to_markdown()
            ),
        }
    )

    return hourly_location




@asset
def rentals_with_weather(context: AssetExecutionContext, hourly_location_rentals: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    weather_features = weather_data.copy().drop(
        columns=["id", "datetime"],
        errors="ignore",
    )

    full_hours = pd.DataFrame(
        {
            "hour": pd.date_range(
                start=hourly_location_rentals["hour"].min(),
                end=hourly_location_rentals["hour"].max(),
                freq="h",
            )
        }
    )

    weather_features = full_hours.merge(
        weather_features,
        on="hour",
        how="left",
    )

    weather_columns = [
        col for col in weather_features.columns
        if col != "hour"
    ]

    # handle missing weather data
    weather_features["is_missing_weather"] = (
        weather_features[weather_columns]
        .isna()
        .any(axis=1)
        .astype(int)
    )

    numeric_weather_cols = (
        weather_features[weather_columns]
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    weather_features[numeric_weather_cols] = (
        weather_features[numeric_weather_cols]
        .interpolate(method="linear")
        .ffill()
        .bfill()
    )

    categorical_weather_cols = (
        weather_features[weather_columns]
        .select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    weather_features[categorical_weather_cols] = (
        weather_features[categorical_weather_cols]
        .ffill()
        .bfill()
    )

    rentals_weather_data = hourly_location_rentals.merge(
        weather_features,
        on="hour",
        how="left",
    )

    context.add_output_metadata(
        {
            "row_count": len(rentals_weather_data),
            "weather_row_count": len(weather_features),
            "is_missing_weather_hours": int(
                weather_features["is_missing_weather"].sum()
            ),
            "remaining_missing_values": int(
                rentals_weather_data.isna().sum().sum()
            ),
            "preview": MetadataValue.md(
                rentals_weather_data.head().to_markdown()
            ),
        }
    )
    return rentals_weather_data


@asset
def final_rental_data(context: AssetExecutionContext, rentals_with_weather: pd.DataFrame, holidays_data: pd.DataFrame) -> pd.DataFrame:
    holidays = holidays_data.copy().drop(
        columns="id",
        errors="ignore",
    )

    final_data = rentals_with_weather.merge(
        holidays[["date", "holiday"]],
        on="date",
        how="left",
    )

    final_data["is_holiday"] = (
        final_data["holiday"].notna().astype(int)
    )

    # add more flags
    final_data["is_workday"] = (
        (final_data["is_weekend"] == 0)
        & (final_data["is_holiday"] == 0)
    ).astype(int)

    # avoid NaN for non-holiday rows
    final_data["holiday"] = final_data["holiday"].fillna("Not a holiday")
    output_path = PROCESSED_DATA_DIR / "final_enriched_rental_data.csv"
    final_data.to_csv(output_path, index=False)

    context.add_output_metadata(
        {
            "output_path": MetadataValue.path(str(output_path.resolve())),
            "row_count": len(final_data),
            "column_count": len(final_data.columns),
            "holiday_rows": int(final_data["is_holiday"].sum()),
            "holiday_dates": int(
                final_data.loc[
                    final_data["is_holiday"] == 1,
                    "date",
                ].nunique()
            ),
            "workday_rows": int(final_data["is_workday"].sum()),
            "remaining_missing_values": int(
                final_data.isna().sum().sum()
            ),
            "preview": MetadataValue.md(
                final_data.head().to_markdown()
            ),
        }
    )

    return final_data


