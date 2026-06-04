
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from dg_pipeline.modeling.baseline import build_preprocessor


def build_random_forest_candidate() -> Pipeline:
    """Build the Random Forest with selected parameters and using baseline features."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )