from pathlib import Path
from typing import ClassVar

import pandas as pd
from dagster import ConfigurableResource


class BikeRentalDataLoader(ConfigurableResource):
    """Load bike rental project source datasets from configurable storage."""

    data_dir: str = "../data"

    CSV_FILES: ClassVar[dict[str, str]] = {
        "registered_rentals": "registered_bike_rentals.csv",
        "direct_pickups": "direct_pickup_bike_rentals.csv",
        "weather_data": "weather.csv",
        "holidays_data": "holidays.csv",
    }

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load one named source dataset from the configured CSV directory."""
        if dataset_name not in self.CSV_FILES:
            raise ValueError(
                f"Unknown dataset name: {dataset_name}. "
                f"Available datasets: {list(self.CSV_FILES)}"
            )

        file_path = Path(self.data_dir) / self.CSV_FILES[dataset_name]

        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found for dataset '{dataset_name}': "
                f"{file_path.resolve()}"
            )

        return pd.read_csv(file_path)