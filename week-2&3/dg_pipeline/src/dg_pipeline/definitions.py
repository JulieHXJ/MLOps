# from pathlib import Path

# from dagster import definitions, load_from_defs_folder


# @definitions
# def defs():
#     return load_from_defs_folder(path_within_project=Path(__file__).parent)


from dagster import Definitions, load_assets_from_modules

from dg_pipeline.defs.assets import raw_data
from dg_pipeline.defs.assets import merged_rentals
from dg_pipeline.defs.assets import enriched_data
from dg_pipeline.defs.assets import baseline_model_training

from dg_pipeline.resources.data_loader import BikeRentalDataLoader
from dg_pipeline.io_managers.csv_io_manager import CsvIOManager


all_assets = load_assets_from_modules(
    [
        raw_data,
        merged_rentals,
        enriched_data,
        baseline_model_training,
    ]
)


defs = Definitions(
    assets=all_assets,
    resources={
        "data_loader": BikeRentalDataLoader(
            data_dir="../data",
        ),
        "io_manager": CsvIOManager(
            base_dir="../data/processed/asset_outputs",
        ),
    },
)