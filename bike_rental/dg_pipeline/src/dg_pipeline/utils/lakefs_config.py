import os
from dataclasses import dataclass

from dotenv import load_dotenv
from lakefs_sdk import Configuration, ApiClient
from lakefs_sdk.api.refs_api import RefsApi


@dataclass(frozen=True)
class LakeFSConfig:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    repo: str
    branch: str
    dataset_path: str
    commit_id: str | None = None



def get_latest_commit_id_from_lakefs_branch(
    config: LakeFSConfig,
) -> str:
    """
    Get the latest commit ID from a LakeFS branch.
    """

    lakefs_configuration = Configuration(
        host=config.endpoint,
        username=config.access_key_id,
        password=config.secret_access_key,
    )

    with ApiClient(lakefs_configuration) as api_client:
        refs_api = RefsApi(api_client)

        commit_list = refs_api.log_commits(
            repository=config.repo,
            ref=config.branch,
            amount=1,
        )

    if not commit_list.results:
        raise ValueError(
            f"No commits found in LakeFS repo='{config.repo}', "
            f"branch='{config.branch}'."
        )

    return commit_list.results[0].id


# def resolve_lakefs_commit_id(
#     config: LakeFSConfig,
# ) -> str:
#     """
#     Resolve the LakeFS commit ID.

#     Priority:
#     1. Use config.commit_id if it is explicitly provided.
#     2. Otherwise, fetch the latest commit ID from the configured branch.
#     """

#     if config.commit_id:
#         return config.commit_id

#     return get_latest_commit_id_from_lakefs_branch(config)

def build_lakefs_metadata(
    config: LakeFSConfig,
) -> dict[str, str]:
    """
    Build LakeFS metadata for MLflow logging.
    """

    lakefs_ref = config.commit_id or config.branch

    metadata = {
        "data_source": "lakefs",
        "lakefs_repo": config.repo,
        "lakefs_branch": config.branch,
        "lakefs_ref": lakefs_ref,
        "lakefs_dataset_path": config.dataset_path,
        "lakefs_uri": (
            f"lakefs://{config.repo}/"
            f"{lakefs_ref}/"
            f"{config.dataset_path}"
        ),
    }

    return {
        key: str(value)
        for key, value in metadata.items()
        if value is not None and value != ""
    }


def get_lakefs_config() -> LakeFSConfig:
    """
    Load LakeFS connection and dataset configuration from environment variables.
    """

    load_dotenv()

    endpoint = os.getenv("LAKEFS_ENDPOINT", "http://127.0.0.1:8000")
    access_key_id = os.getenv("LAKEFS_ACCESS_KEY_ID", "")
    secret_access_key = os.getenv("LAKEFS_SECRET_ACCESS_KEY", "")
    repo = os.getenv("LAKEFS_REPO", "bike-rental")
    branch = os.getenv("LAKEFS_BRANCH", "main")
    dataset_path = os.getenv(
        "LAKEFS_DATASET_PATH",
        "data/processed/engineered_model_data.csv",
    )
    commit_id = os.getenv("LAKEFS_COMMIT_ID") or None

    if not access_key_id or not secret_access_key:
        raise ValueError(
            "LakeFS credentials are missing. Please set "
            "LAKEFS_ACCESS_KEY_ID and LAKEFS_SECRET_ACCESS_KEY in .env."
        )

    return LakeFSConfig(
        endpoint=endpoint,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        repo=repo,
        branch=branch,
        dataset_path=dataset_path,
        commit_id=commit_id,
    )
