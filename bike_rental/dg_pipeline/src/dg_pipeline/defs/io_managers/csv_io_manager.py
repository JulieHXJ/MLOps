from pathlib import Path

import pandas as pd
from dagster import ConfigurableIOManager, InputContext, OutputContext

class CsvIOManager(ConfigurableIOManager):
    """
    Save pandas DataFrame and Series assets as CSV files.

    DataFrame assets are saved directly.
    Series assets are converted to a single-column CSV and restored as Series
    when loaded.
    """

    base_dir: str = "processed/asset_outputs"

    def _get_path(self, context: OutputContext | InputContext) -> Path:
        asset_key = context.asset_key

        if asset_key is None:
            raise ValueError("CsvIOManager requires an asset key.")

        asset_name = "_".join(asset_key.path)
        return Path(self.base_dir) / f"{asset_name}.csv"

    def handle_output(self, context: OutputContext, obj) -> None:
        path = self._get_path(context)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(obj, pd.DataFrame):
            obj.to_csv(path, index=False)
            context.add_output_metadata(
                {
                    "path": str(path.resolve()),
                    "row_count": len(obj),
                    "column_count": len(obj.columns),
                    "stored_type": "DataFrame",
                }
            )
            return
        if isinstance(obj, pd.Series):
            series_name = obj.name or "value"

            obj.to_frame(name=series_name).to_csv(path, index=False)

            context.add_output_metadata(
                {
                    "path": str(path.resolve()),
                    "row_count": len(obj),
                    "column_count": 1,
                    "stored_type": "Series",
                    "series_name": series_name,
                }
            )
            return
        raise TypeError(
            "CsvIOManager only supports pandas DataFrame and Series outputs. "
            f"Asset '{context.asset_key}' returned {type(obj).__name__}."
        )

    def load_input(self, context: InputContext):
        path = self._get_path(context)

        if not path.exists():
            raise FileNotFoundError(
                f"Expected asset file does not exist: {path.resolve()}"
            )

        data = pd.read_csv(path)

        stored_type = None
        if context.upstream_output is not None:
            stored_type = context.upstream_output.metadata.get("stored_type")

            if hasattr(stored_type, "value"):
                stored_type = stored_type.value

        if stored_type == "Series":
            return data.iloc[:, 0]

        return data