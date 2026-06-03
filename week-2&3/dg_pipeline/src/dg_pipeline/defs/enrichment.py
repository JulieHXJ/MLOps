from pathlib import Path

import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset


DATA_DIR = Path("../data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

@asset(group_name="enrichment")
def rentals_with_weather(context: AssetExecutionContext, hourly_location_rentals: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    """Merge observed hourly rental demand with available weather observations."""
    weather_features = weather_data.copy().drop(
        columns=["id", "datetime"],
        errors="ignore",
    )

    duplicate_weather_hours = int(
        weather_features["hour"].duplicated().sum()
    )

    if duplicate_weather_hours > 0:
        raise ValueError(
            "Weather data contains "
            f"{duplicate_weather_hours} duplicate hourly records."
        )
    
    rentals_weather_data = hourly_location_rentals.merge(
        weather_features,
        on="hour",
        how="left",
    )

    remaining_missing_values = int(
        rentals_weather_data.isna().sum().sum()
    )

    if remaining_missing_values > 0:
        raise ValueError(
            "Weather values are missing for observed rental hours."
        )


    context.add_output_metadata(
        {
            "row_count": len(rentals_weather_data),
            "weather_rows_used": int(
                rentals_weather_data["hour"].nunique()
            ),
            "remaining_missing_values": remaining_missing_values,
            "preview": MetadataValue.md(
                rentals_weather_data.head().to_markdown()
            ),
        }
    )
    return rentals_weather_data


@asset(group_name="enrichment")
def final_enriched_rental_data(context: AssetExecutionContext, rentals_with_weather: pd.DataFrame, holidays_data: pd.DataFrame) -> pd.DataFrame:
    """Add holiday information and derive holiday/workday indicators."""
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
