from pathlib import Path
from typing import Any

import dagster as dg
import pandas as pd


class CsvIOManager(dg.ConfigurableIOManager):
    """Store Pandas DataFrame asset outputs as CSV files and load them downstream."""

    base_dir: str = "processed/asset_outputs"

    def _get_path(self, asset_key: dg.AssetKey) -> Path:
        """Build a CSV file path from a Dagster asset key."""
        path = Path(self.base_dir).joinpath(*asset_key.path)
        return path.with_suffix(".csv")

    def handle_output(
        self,
        context: dg.OutputContext,
        obj: Any,
    ) -> None:
        """Write one asset output DataFrame to a CSV file."""
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(
                "CsvDataFrameIOManager only supports pandas DataFrame outputs. "
                f"Asset '{context.asset_key.to_user_string()}' returned "
                f"{type(obj).__name__}."
            )

        output_path = self._get_path(context.asset_key)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        obj.to_csv(output_path, index=False)

        context.add_output_metadata(
            {
                "csv_storage_path": dg.MetadataValue.path(
                    str(output_path.resolve())
                ),
                "stored_row_count": len(obj),
                "stored_column_count": len(obj.columns),
            }
        )

    def load_input(
        self,
        context: dg.InputContext,
    ) -> pd.DataFrame:
        """Load an upstream asset DataFrame from its CSV file."""
        if context.upstream_output is None:
            raise ValueError(
                "CsvDataFrameIOManager can only load inputs from upstream assets."
            )

        input_path = self._get_path(
            context.upstream_output.asset_key
        )

        if not input_path.exists():
            raise FileNotFoundError(
                "Stored CSV output was not found for upstream asset "
                f"'{context.upstream_output.asset_key.to_user_string()}': "
                f"{input_path.resolve()}"
            )

        data = pd.read_csv(input_path)

        datetime_columns = [
            column
            for column in ["datetime", "hour", "date"]
            if column in data.columns
        ]

        for column in datetime_columns:
            data[column] = pd.to_datetime(data[column])

        return data