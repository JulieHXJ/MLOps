import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset

from dg_pipeline.utils.lakefs_config import get_lakefs_config
from dg_pipeline.utils.lakefs_io import build_lakefs_uri, read_csv_from_lakefs


@asset(group_name="lakefs_data")
def lakefs_engineered_model_data(
    context: AssetExecutionContext,
) -> pd.DataFrame:
    """
    Load the production model-ready engineered dataset from LakeFS.
    """

    config = get_lakefs_config()
    data = read_csv_from_lakefs(config)
    lakefs_uri = build_lakefs_uri(config)

    context.add_output_metadata(
        {
            "data_source": "lakefs",
            "lakefs_uri": lakefs_uri,
            "lakefs_repo": config.repo,
            "lakefs_branch": config.branch,
            "lakefs_dataset_path": config.dataset_path,
            "lakefs_commit_id": config.commit_id or "not_set",
            "rows": len(data),
            "columns": len(data.columns),
            "preview": MetadataValue.md(data.head().to_markdown()),
        }
    )

    return data