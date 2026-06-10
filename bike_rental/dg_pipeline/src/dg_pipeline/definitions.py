# from pathlib import Path

# from dagster import definitions, load_from_defs_folder


# @definitions
# def defs():
#     return load_from_defs_folder(path_within_project=Path(__file__).parent)


from dagster import Definitions, load_assets_from_package_module

from dg_pipeline.defs import assets as assets_package
from dg_pipeline.defs.resources.data_loader import BikeRentalDataLoader
from dg_pipeline.defs.io_managers.csv_io_manager import CsvIOManager



from dg_pipeline.defs.assets.data.lakefs_data import (
    lakefs_engineered_model_data_not_empty,
    lakefs_engineered_model_data_required_columns,
    lakefs_engineered_model_data_target_not_missing,
    lakefs_engineered_model_data_target_non_negative,
    lakefs_engineered_model_data_hour_parseable,
)

all_assets = load_assets_from_package_module(
    assets_package
)


all_asset_checks = [
    lakefs_engineered_model_data_not_empty,
    lakefs_engineered_model_data_required_columns,
    lakefs_engineered_model_data_target_not_missing,
    lakefs_engineered_model_data_target_non_negative,
    lakefs_engineered_model_data_hour_parseable,
]

defs = Definitions(
    assets=all_assets,
    asset_checks=all_asset_checks,
    resources={
        "data_loader": BikeRentalDataLoader(
            data_dir="../data",
        ),
        "io_manager": CsvIOManager(
            base_dir="../data/processed/asset_outputs",
        ),
    },
)