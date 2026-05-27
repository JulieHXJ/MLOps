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
def hourly_rentals(context: AssetExecutionContext, registered_rentals: pd.DataFrame, direct_pickups: pd.DataFrame) -> pd.DataFrame:
    # aggregate hourly registered rentals
    registered_hourly = (registered_rentals.groupby("hour").size().reset_index(name="registered_count"))

    # aggregate hourly pickups
    direct_hourly = (direct_pickups.groupby("hour").size().reset_index(name="direct_count"))

    # merge
    rentals = registered_hourly.merge(direct_hourly, on="hour", how="outer")
    rentals = rentals.sort_values("hour")

    # mark missing hours and fill up with 0
    full_hours = pd.DataFrame({"hour": pd.date_range(start=rentals["hour"].min(), end=rentals["hour"].max(), freq="h")})
    rentals = full_hours.merge(rentals, on="hour", how="left")
    rentals["missing_rental"] = (rentals["registered_count"].isna() & rentals["direct_count"].isna()).astype(int)

    rentals["registered_count"] = rentals["registered_count"].fillna(0).astype(int)
    rentals["direct_count"] = rentals["direct_count"].fillna(0).astype(int)
    rentals["total_count"] = (rentals["direct_count"] + rentals["registered_count"])

    # add time features like hour_of_day, weekday, is_weekend
    rentals["date"] = rentals["hour"].dt.date
    rentals["hour_of_day"] = rentals["hour"].dt.hour
    rentals["day_of_week"] = rentals["hour"].dt.weekday
    rentals["is_weekend"] = rentals["day_of_week"].isin([5, 6]).astype(int)

    # check summary in Dagster UI
    context.add_output_metadata(
        {
            "hourly_rental_row_count": len(rentals),
            "registered_count": len(registered_rentals),
            "direct_count": len(direct_pickups),
            "expected_total_count": len(registered_rentals) + len(direct_pickups),
            "total_count_sum": int(rentals["total_count"].sum()),
            "missing_rental_hours": int(rentals["missing_rental"].sum()),
            "preview": MetadataValue.md(rentals.head().to_markdown()),
        }
    )

    return rentals




@asset
def rentals_with_weather(context: AssetExecutionContext, hourly_rentals: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    weather = weather_data.copy().drop(columns=["id", "datetime"], errors="ignore")
    weather = hourly_rentals.merge(weather, on="hour", how="left")
    weather = weather.sort_values("hour")

    # handle missing values
    weather_col = [
        col for col in weather_data.columns
        if col not in [
            "id",
            "datetime"
        ]
    ]

    weather["missing_weather"] = weather[weather_col].isna().all(axis=1).astype(int)

    # numeric
    numeric_col = weather[weather_col].select_dtypes(include="number").columns
    weather[numeric_col] = (weather[numeric_col].interpolate(method="linear", limit_direction="both"))

    # categorical
    categorical_col = [
        col for col in weather_col
        if col not in numeric_col
    ]
    weather[categorical_col] = (weather[categorical_col].ffill().bfill())

    context.add_output_metadata(
        {
            "row_count": len(weather),
            "missing_weather_count": int(weather["missing_weather"].sum()),
            "remaining_missing_values": int(weather.isna().sum().sum()),
            "preview": MetadataValue.md(weather.head().to_markdown()),
        }
    )

    return weather


@asset
def final_rental_data(context: AssetExecutionContext, rentals_with_weather: pd.DataFrame, holidays_data: pd.DataFrame) -> pd.DataFrame:
    final_data = holidays_data.copy().drop(columns="id", errors="ignore")
    final_data = rentals_with_weather.merge(final_data, on="date", how="left")
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
    data = final_rental_data.copy().drop(columns="holiday", errors="ignore")
    out_path = PROCESSED_DATA_DIR/"final_rental_data_dg.scv"
    data.to_csv(out_path, index=False)
    return str(out_path)

