import pandas as pd
from pathlib import Path
from dagster import asset, AssetExecutionContext, MetadataValue


DATA_DIR = Path("../data")
# output file path
PROCESSED_DATA_DIR = Path("processed")
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

    hourly_location["total_count"] = (hourly_location["registered_count"] + hourly_location["direct_count"])
    hourly_location = hourly_location.sort_values(["hour", "location_id"]).reset_index(drop=True)

    # create full hour and location tables 
    full_hours = pd.DataFrame({"hour": pd.date_range(start=hourly_location["hour"].min(), end=hourly_location["hour"].max(),freq="h")})
    all_locations = pd.DataFrame({"location_id": all_rental_events["location_id"].drop_duplicates().sort_values()})

    full_table = full_hours.merge(all_locations, how="cross")
    hourly_location = full_table.merge(hourly_location, on=["hour", "location_id"], how="left")

    #handle missing values
    hourly_location["is_missing_rental"] = (hourly_location["total_count"].isna().astype(int))

    count_cols = ["registered_count", "direct_count", "total_count"]
    hourly_location[count_cols] = (hourly_location[count_cols].fillna(0).astype(int))

    # add time features like hour_of_day, weekday, is_weekend
    hourly_location["date"] = hourly_location["hour"].dt.date
    hourly_location["hour_of_day"] = hourly_location["hour"].dt.hour
    hourly_location["day_of_week"] = hourly_location["hour"].dt.dayofweek
    hourly_location["month"] = hourly_location["hour"].dt.month
    hourly_location["is_weekend"] = (hourly_location["day_of_week"].isin([5, 6]).astype(int))

    # check summary in Dagster UI
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
            "is_missing_rental_rows": int(
                hourly_location["is_missing_rental"].sum()
            ),
            "preview": MetadataValue.md(hourly_location.head().to_markdown()),
        }
    )

    return hourly_location




@asset
def rentals_with_weather(context: AssetExecutionContext, hourly_location_rentals: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    weather_features = weather_data.copy().drop(columns=["id", "datetime"], errors="ignore")

    # handle missing weather date
    full_table = pd.DataFrame({
        "hour": pd.date_range(
            start=hourly_location_rentals["hour"].min(),
            end=hourly_location_rentals["hour"].max(),
            freq="h",
        )
    })

    weather_features = full_table.merge(weather_features, on="hour", how="left")
    weather_columns = [
        col for col in weather_features.columns
        if col != "hour"
    ]

    weather_features["is_missing_weather"] = (weather_features[weather_columns].isna().any(axis=1).astype(int))
    
    numeric_weather_cols = (weather_features[weather_columns].select_dtypes(include="number").columns.tolist())
    weather_features[numeric_weather_cols] = (weather_features[numeric_weather_cols].interpolate(method="linear").ffill().bfill())

    categorical_weather_cols = (weather_features[weather_columns].select_dtypes(exclude="number").columns.tolist())
    weather_features[categorical_weather_cols] = (weather_features[categorical_weather_cols].ffill().bfill())
    
    # merge
    weather = hourly_location_rentals.merge(weather_features, on="hour", how="left")

    context.add_output_metadata(
        {
            "row_count": len(weather),
            "weather_row_count": len(weather_features),
            "is_missing_weather_hours": int(
                weather_features["is_missing_weather"].sum()
            ),
            "remaining_missing_values": int(weather.isna().sum().sum()),
            "preview": MetadataValue.md(weather.head().to_markdown()),
        }
    )
    return weather


@asset
def final_rental_data(context: AssetExecutionContext, rentals_with_weather: pd.DataFrame, holidays_data: pd.DataFrame) -> pd.DataFrame:
    holidays = holidays_data.copy().drop(columns="id", errors="ignore")
    final_data = rentals_with_weather.merge(holidays[["date", "holiday"]], on="date", how="left")
    final_data["is_holiday"] = final_data["holiday"].notna().astype(int)

    context.add_output_metadata(
        {
            "row_count": len(final_data),
            "holiday_rows": int(final_data["is_holiday"].sum()),
            "holiday_dates": int(final_data.loc[final_data["is_holiday"], "date"].nunique()),
            "preview": MetadataValue.md(final_data.head().to_markdown()),
        }
    )

    return final_data

@asset
def final_model_csv(final_rental_data: pd.DataFrame) -> str:
    full_data = final_rental_data.copy().drop(columns="holiday", errors="ignore")
    model_data = full_data.drop(columns=["holiday"], errors="ignore")

    full_output_path = PROCESSED_DATA_DIR / "final_data_with_holiday.csv"
    model_output_path = PROCESSED_DATA_DIR / "final_model_data.csv"

    full_data.to_csv(full_output_path, index=False)
    model_data.to_csv(model_output_path, index=False)

    return str(model_output_path)

