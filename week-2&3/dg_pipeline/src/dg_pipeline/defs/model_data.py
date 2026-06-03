from pathlib import Path

import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset

from dg_pipeline.transformations.model import (
    aggregate_city_hourly_demand,
    validate_hourly_feature_consistency,
    validate_model_ready_data,
)


DATA_DIR = Path("../data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)



@asset(group_name="model_data")
def model_ready_data(context: AssetExecutionContext, final_enriched_rental_data: pd.DataFrame) -> pd.DataFrame:
    """Create the final hourly dataset for model training."""
    data = final_enriched_rental_data.copy()
    data["hour"] = pd.to_datetime(data["hour"])
    data["date"] = pd.to_datetime(data["date"])

    validate_hourly_feature_consistency(data)

    hourly_data = aggregate_city_hourly_demand(data)

    validation_metadata = validate_model_ready_data(hourly_data)

    output_path = (PROCESSED_DATA_DIR / "model_ready_hourly_rental_data.csv")

    hourly_data.to_csv(output_path, index=False)



    context.add_output_metadata(
        {
            "output_path": MetadataValue.path(str(output_path.resolve())),
            "input_location_level_rows": len(data),
            "final_hourly_rows": len(hourly_data),
            "final_column_count": len(hourly_data.columns),
            **validation_metadata,
            "total_direct_rentals": int(
                hourly_data["direct_count"].sum()
            ),
            "total_registered_rentals": int(
                hourly_data["registered_count"].sum()
            ),
            "total_rentals": int(
                hourly_data["total_count"].sum()
            ),
            "preview": MetadataValue.md(
                hourly_data.head().to_markdown()
            ),
        }
    )


    return hourly_data

