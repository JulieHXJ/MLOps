import pandas as pd


def aggregate_rental_counts(all_rental_events: pd.DataFrame) -> pd.DataFrame:
    """Count registered and direct rentals for each observed hour and location."""
    hourly_location = (
        all_rental_events
        .groupby(["hour", "location_id", "rental_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

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

    return hourly_location
    


# def complete_hour_location_grid(hourly_location: pd.DataFrame, all_rental_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """Create a complete hour-location grid and mark artificially filled rows."""
#     full_hours = pd.DataFrame(
#         {
#             "hour": pd.date_range(
#                 start=hourly_location["hour"].min(),
#                 end=hourly_location["hour"].max(),
#                 freq="h",
#             )
#         }
#     )

#     all_locations = pd.DataFrame(
#         {
#             "location_id": (
#                 all_rental_events["location_id"]
#                 .drop_duplicates()
#                 .sort_values()
#             )
#         }
#     )

#     full_table = full_hours.merge(all_locations, how="cross")

#     hourly_location = full_table.merge(
#         hourly_location,
#         on=["hour", "location_id"],
#         how="left",
#     )

#     hourly_location["is_zero_filled_rental"] = (
#         hourly_location["total_count"].isna().astype(int)
#     )

#     count_cols = [
#         "registered_count",
#         "direct_count",
#         "total_count",
#     ]

#     hourly_location[count_cols] = (
#         hourly_location[count_cols]
#         .fillna(0)
#         .astype(int)
#     )

#     return hourly_location, all_locations, full_table



def add_time_features(hourly_location: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from the hourly timestamp column."""
    hourly_location = hourly_location.copy()

    hourly_location["date"] = hourly_location["hour"].dt.date
    hourly_location["hour_of_day"] = hourly_location["hour"].dt.hour
    hourly_location["day_of_week"] = hourly_location["hour"].dt.dayofweek
    hourly_location["month"] = hourly_location["hour"].dt.month
    hourly_location["is_weekend"] = (
        hourly_location["day_of_week"].isin([5, 6]).astype(int)
    )

    return hourly_location