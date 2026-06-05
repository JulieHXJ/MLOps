import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset


def combine_rental_events(
    registered_rentals: pd.DataFrame,
    direct_pickups: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine registered rentals and direct pickups into one event-level table.
    """

    registered_events = registered_rentals[["hour", "location_id"]].copy()
    registered_events["rental_type"] = "registered"

    direct_events = direct_pickups[["hour", "location_id"]].copy()
    direct_events["rental_type"] = "direct"

    rental_events = pd.concat(
        [registered_events, direct_events],
        ignore_index=True,
    )

    return rental_events


def aggregate_rental_counts(rental_events: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate rental events into hourly location-level rental demand.
    """

    hourly_location_rentals = (
        rental_events
        .groupby(["hour", "location_id", "rental_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    hourly_location_rentals = hourly_location_rentals.rename(
        columns={
            "registered": "registered_count",
            "direct": "direct_count",
        }
    )

    if "registered_count" not in hourly_location_rentals.columns:
        hourly_location_rentals["registered_count"] = 0

    if "direct_count" not in hourly_location_rentals.columns:
        hourly_location_rentals["direct_count"] = 0

    hourly_location_rentals["total_count"] = (
        hourly_location_rentals["registered_count"]
        + hourly_location_rentals["direct_count"]
    )

    return hourly_location_rentals



def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic time-based features from the hourly timestamp.
    """

    data = data.copy()

    data["hour"] = pd.to_datetime(data["hour"])

    data["date"] = data["hour"].dt.date
    data["hour_of_day"] = data["hour"].dt.hour
    data["day_of_week"] = data["hour"].dt.weekday
    data["month"] = data["hour"].dt.month
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    return data


@asset(group_name="merged_rentals")
def hourly_location_rentals(
    context: AssetExecutionContext,
    registered_rentals: pd.DataFrame,
    direct_pickups: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate all rental events into hourly location-level demand.

    This asset does not fill missing hours.
    It only keeps observed rental hours from the original data.
    """

    rental_events = combine_rental_events(
        registered_rentals=registered_rentals,
        direct_pickups=direct_pickups,
    )

    hourly_location = aggregate_rental_counts(rental_events)
    hourly_location = add_time_features(hourly_location)

    context.add_output_metadata(
        {
            "row_count": len(hourly_location),
            "observed_hours": int(hourly_location["hour"].nunique()),
            "location_count": int(hourly_location["location_id"].nunique()),
            "registered_count_sum": int(hourly_location["registered_count"].sum()),
            "direct_count_sum": int(hourly_location["direct_count"].sum()),
            "total_count_sum": int(hourly_location["total_count"].sum()),
            "preview": MetadataValue.md(hourly_location.head().to_markdown()),
        }
    )

    return hourly_location