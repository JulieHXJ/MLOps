import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset

from dg_pipeline.transformations.rental import (
    add_time_features,
    aggregate_rental_counts,
)

@asset(group_name="rental_processing")
def all_rental_events(registered_rentals: pd.DataFrame, direct_pickups: pd.DataFrame) -> pd.DataFrame:
    """Combine registered rentals and direct pickups into one event-level table."""
    registered_events = registered_rentals[["hour", "location_id"]].copy()
    registered_events["rental_type"] = "registered"

    direct_events = direct_pickups[["hour", "location_id"]].copy()
    direct_events["rental_type"] = "direct"

    all_events = pd.concat([registered_events, direct_events], ignore_index=True)
    return all_events




@asset(group_name="rental_processing")
def hourly_location_rentals(context: AssetExecutionContext, all_rental_events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rental events into hourly location-level demand with time features."""
    hourly_location = aggregate_rental_counts(all_rental_events)
    hourly_location = add_time_features(hourly_location)

    context.add_output_metadata(
        {
            "row_count": len(hourly_location),
            "observed_hours": int(hourly_location["hour"].nunique()),
            "location_count": int(
                hourly_location["location_id"].nunique()
            ),
            "registered_count_sum": int(
                hourly_location["registered_count"].sum()
            ),
            "direct_count_sum": int(
                hourly_location["direct_count"].sum()
            ),
            "total_count_sum": int(
                hourly_location["total_count"].sum()
            ),
            "preview": MetadataValue.md(
                hourly_location.head().to_markdown()
            ),
        }
    )

    return hourly_location