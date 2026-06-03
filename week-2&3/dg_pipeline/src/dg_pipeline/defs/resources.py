from dagster import Definitions, definitions

from dg_pipeline.io_managers.csv_io_manager import CsvIOManager
from dg_pipeline.resources.data_loader import BikeRentalDataLoader


@definitions
def resources() -> Definitions:
    """Provide configured resources for the bike rental pipeline."""
    return Definitions(
        resources={
            "data_loader": BikeRentalDataLoader(
                data_dir="../data",
            ),
            "io_manager": CsvIOManager(
                base_dir="processed/asset_outputs",
            ),
        },
    )