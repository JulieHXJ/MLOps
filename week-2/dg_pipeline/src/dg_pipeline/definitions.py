from pathlib import Path

from dagster import definitions, load_from_defs_folder, load_assets_from_modules


@definitions
def defs():
    return load_from_defs_folder(path_within_project=Path(__file__).parent)
