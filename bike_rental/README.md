# Bike Rental Demand Forecasting MLOps Pipeline
## Project Overview
This project implements an MLOps pipeline for bike rental demand forecasting. The pipeline starts from raw bike rental, weather, and holiday data, transforms them into hourly model-ready datasets, trains several regression models, tracks experiments with MLflow, versions datasets with LakeFS, and registers the selected production model in the MLflow Model Registry.

The final production model is an XGBoost regression model trained on engineered hourly demand features.

## Pipeline Overview

The project contains three main stages:

Data processing
    ↓
Model training and comparison
    ↓
MLOps integration with MLflow and LakeFS

The main processed datasets are:

- enriched_rental_data: rental data enriched with weather and holiday information.
- aggregated_hourly_data: city-level hourly rental demand dataset.
- engineered_model_data: final model-ready dataset with engineered time, calendar, cyclical, lag, and rolling demand features.

The production model is trained on: `engineered_model_data.csv`

## Environment Setup
The main MLOps services used in the project are:

- Dagster for workflow orchestration
- MLflow for experiment tracking and model registry
- LakeFS for data versioning
- pandas, scikit-learn, and XGBoost for data processing and model training

### Install Dependencies
```
uv sync
uv add mlflow lakefs lakefs-spec python-dotenv xgboost
```

### env
Create a local .env file. A safe template can be stored as `.env.example`

## Run services
### 1. LakeFS
```
uv run python -m lakefs.quickstart
```
The LakeFS UI is available at:
```
http://127.0.0.1:8000
```

### 2. MLflow
```
uv run mlflow ui --port 5000
```
The MLflow UI is available at:

```
http://127.0.0.1:5000
```
### 3. Dagster
```
uv run dg dev
```
The Dagster UI is available at the local URL shown in the terminal.



## Data Processing Summary

The pipeline is organized as a sequence of Dagster assets. The raw rental assets are first transformed into hourly rental demand, then enriched with weather and holiday information, and finally exported as a model-ready CSV file.

```text
registered_rentals ─┐
                    ├──> hourly_location_rentals ─┐
direct_pickups ─────┘                              │
                                                   ├──> rentals_with_weather ─┐
weather_data ──────────────────────────────────────┘                          │
                                                                              ├──> enriched_rental_data ───> engineered_model_data
holidays_data ────────────────────────────────────────────────────────────────┘
```

The engineered feature dataset includes:

- calendar features
- rush hour indicators
- weekend and holiday indicators
- cyclical encodings for hour, weekday, and month
- lag features
- rolling demand features
- weather variables
- categorical weather condition features


## Dagster Asset Checks

Before model training, the LakeFS dataset is validated with Dagster asset checks.

The checks verify that:

- the dataset is not empty
- all required engineered feature columns are present
- the target column has no missing values
- the target column has no negative values
- the hourly timestamp column can be parsed correctly

These checks help prevent invalid or unexpected data from entering the model training workflow.


## Model Training

The project compares multiple regression models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

## MLflow Experiment Tracking

MLflow is used to track model training runs. Each run logs:

- model name
- feature set
- model stage
- model parameters
- train/test metrics
- trained model artifact
- dataset preview
- LakeFS dataset metadata

## LakeFS Data Versioning Strategy
Each MLflow model run records:

- lakefs_repo
- lakefs_branch
- lakefs_commit_id
- lakefs_dataset_path
- lakefs_uri

## Future Improvements

Possible future improvements include:

- Compare chronological split and random split strategies.
- Add automated tests for data loading and feature engineering.
- Add a model promotion rule instead of manually selecting the production model.
- Add a LakeFS-backed IO manager for more complete asset versioning.