# from pathlib import Path

# from dagster import definitions, load_from_defs_folder


# @definitions
# def defs():
#     return load_from_defs_folder(path_within_project=Path(__file__).parent)


from dagster import Definitions, load_assets_from_package_module

from dg_pipeline.defs import assets as assets_package
from dg_pipeline.defs.resources.data_loader import BikeRentalDataLoader
from dg_pipeline.defs.io_managers.csv_io_manager import CsvIOManager


all_assets = load_assets_from_package_module(
    assets_package
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