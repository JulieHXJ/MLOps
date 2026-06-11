import pandas as pd

from io import StringIO
from lakefs_spec import LakeFSFileSystem

from dg_pipeline.utils.lakefs_config import LakeFSConfig


def build_lakefs_uri(config: LakeFSConfig) -> str:
    """
    Build a lakeFS URI for the configured dataset.
    """

    ref = config.commit_id or config.branch
    return f"lakefs://{config.repo}/{ref}/{config.dataset_path}"


def read_csv_from_lakefs(config: LakeFSConfig) -> pd.DataFrame:
    """
    Read a CSV dataset from LakeFS using lakefs-spec.
    """

    fs = LakeFSFileSystem(
        host=config.endpoint,
        username=config.access_key_id,
        password=config.secret_access_key,
    )

    lakefs_uri = build_lakefs_uri(config)

    with fs.open(lakefs_uri, "rb") as file:
        data = pd.read_csv(file)

    return data


def write_csv_to_lakefs(
    data: pd.DataFrame,
    config,
) -> None:
    """
    Write a pandas DataFrame as a CSV object to LakeFS.
    """

    fs = LakeFSFileSystem(
        host=config.endpoint,
        username=config.access_key_id,
        password=config.secret_access_key,
    )

    lakefs_uri = build_lakefs_uri(config)

    with fs.open(lakefs_uri, "wb") as file:
        data.to_csv(file, index=False)