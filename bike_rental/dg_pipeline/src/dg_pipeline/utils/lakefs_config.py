import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class LakeFSConfig:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    repo: str
    branch: str
    dataset_path: str
    commit_id: str | None = None


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
    commit_id = os.getenv("LAKEFS_COMMIT_ID")

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