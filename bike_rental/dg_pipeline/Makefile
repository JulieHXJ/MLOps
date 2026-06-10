.PHONY: install lakefs mlflow dagster api check-xgboost

LIBOMP_PATH := /Users/XiaojunHuang/homebrew/opt/libomp/lib

install:
	uv sync

lakefs:
	uv run python -m lakefs.quickstart

mlflow:
	uv run mlflow ui --port 5000

dagster:
	DYLD_LIBRARY_PATH=$(LIBOMP_PATH):$$DYLD_LIBRARY_PATH uv run dg dev

check-xgboost:
	DYLD_LIBRARY_PATH=$(LIBOMP_PATH):$$DYLD_LIBRARY_PATH uv run python -c "from xgboost import XGBRegressor; print('xgboost ok')"

api:
	DYLD_LIBRARY_PATH=$(LIBOMP_PATH):$$DYLD_LIBRARY_PATH uv run uvicorn dg_pipeline.api.main:app --reload --port 8001