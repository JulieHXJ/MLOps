# Bike Rental Demand Forecasting MLOps Pipeline

## Project Overview

This project implements an MLOps pipeline for bike rental demand forecasting. The pipeline starts from raw bike rental, weather, and holiday data, transforms them into hourly model-ready datasets, trains several regression models, tracks experiments with MLflow, versions datasets with LakeFS, and registers the selected production model in the MLflow Model Registry.

The current production model is selected automatically from the engineered model comparison table based on the lowest test RMSE. The selected model is registered in MLflow and assigned the `production` alias.

## Pipeline Overview

The project contains four main stages:

```text
Data processing
    ↓
Feature engineering
    ↓
Model training and comparison
    ↓
MLOps integration with MLflow, LakeFS, and FastAPI
```



## Environment Setup

The main MLOps services used in the project are:

* Dagster for workflow orchestration
* MLflow for experiment tracking and model registry
* LakeFS for data versioning
* FastAPI for model serving
* pandas, scikit-learn, and XGBoost for data processing and model training

### Install Dependencies

```bash
uv sync
uv add mlflow lakefs lakefs-spec python-dotenv xgboost fastapi uvicorn
```

### Environment Variables

Create a local `.env` file. A safe template can be stored as `.env.example`.


## Run Services

Common service commands are managed through the project `Makefile`.

The Makefile includes commands for starting:

- LakeFS
- MLflow
- Dagster
- FastAPI

Example usage:

```bash
make lakefs
make mlflow
make dagster
make api
```

The LakeFS UI is available at:

```text
http://127.0.0.1:8000
```

The MLflow UI is available at:

```text
http://127.0.0.1:5000
```

The FastAPI Swagger UI is available at:

```text
http://127.0.0.1:8000/docs
```

## Data Processing Summary

The pipeline is organized as a sequence of Dagster assets. The raw rental assets are first transformed into hourly rental demand, then enriched with weather and holiday information, then aggregated into city-level hourly demand, and finally transformed into engineered model data.

```text
registered_rentals ─┐
                    ├──> hourly_location_rentals ─┐
direct_pickups ─────┘                              │
                                                   ├──> rentals_with_weather ─┐
weather_data ──────────────────────────────────────┘                          │
                                                                              ├──> enriched_rental_data
holidays_data ────────────────────────────────────────────────────────────────┘
                                                                                      ↓
                                                                            aggregated_hourly_data
                                                                                      ↓
                                                                            engineered_model_data
                                                                                      ↓
                                                                            historical_demand.csv
```

The engineered feature dataset includes:

- calendar features
- rush hour indicators
- weekend, holiday, and workday indicators
- cyclical encodings for hour, weekday, and month
- lag features
- rolling demand features
- same-hour same-weekday historical demand features
- weather variables
- categorical weather condition features

## Model Training

The project compares multiple regression models:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

The main feature sets are:

* `baseline`: basic time, weather, holiday, and condition features
* `engineered`: calendar, cyclical, lag, rolling, weather, and categorical features

The engineered feature set is used for production model selection.

## Split Strategy

The current production model selection uses a chronological 80/20 train/test split.

The pipeline has also been refactored to support multiple split strategies:

* `chronological / time-base`
* `random`

The random split option has been added as an experimental extension point, but it has not yet been fully tested or used for production model selection.

For forecasting-style demand prediction, chronological split remains the preferred strategy because it better simulates predicting future demand from past data.

## Automatic Production Model Registration

The production model is selected automatically from the engineered model comparison table. This means the API does not need to know which model version is currently best. It always loads the model version assigned to the `production` alias.

The workflow is:

```text
engineered_model_comparison
        ↓
select model with lowest test_rmse
        ↓
read corresponding MLflow run_id
        ↓
register runs:/<run_id>/model
        ↓
set alias production
```


## FastAPI Prediction API
The API is designed to accept raw, user-friendly input instead of requiring users to manually provide engineered features.

### Prediction Input

The `/predict` endpoint accepts:

```json
{
  "timestamp": "2024-06-01T08:00:00",
  "temperature_c": 18.5,
  "humidity": 65.0,
  "windspeed_kmh": 12.0,
  "conditions": "Clear",
  "is_holiday": 0
}
```

The API internally generates:

* weekend flag
* workday flag
* rush hour flag
* cyclical time features
* season
* 24-hour lag demand
* 7-day lag demand
* 24-hour rolling mean demand
* 7-day rolling mean demand
* previous 4-week same-hour same-weekday demand mean

These historical demand features are generated from:

```text
historical_demand.csv
```

### API Endpoints

| Endpoint         | Method | Description                                                      |
| ---------------- | ------ | ---------------------------------------------------------------- |
| `/`              | GET    | Root endpoint with available routes                              |
| `/health`        | GET    | Health check endpoint                                            |
| `/model-info`    | GET    | Shows model URI, MLflow tracking URI, and historical data status |
| `/predict`       | POST   | Predicts bike rental demand from raw user-friendly input         |
| `/predict-batch` | POST   | Predicts bike rental demand for multiple raw input records       |


## Current Limitations

* The API depends on a local historical demand CSV file.
* LakeFS is integrated as a versioned dataset source, but a full LakeFS IO manager has not yet been implemented.
* The pipeline can resolves the latest existing commit ID from the configured LakeFS branch, but does not yet automatically create a new LakeFS commit after uploading updated data.
* The API currently assumes the historical demand file contains the required lag and rolling windows for the requested timestamp.

## Future Improvements

* Fully test and compare chronological split and random split strategies.
* Replace the local historical demand CSV with a database, feature store, or LakeFS-backed historical feature source.
* Automatically retrieve the current LakeFS commit ID instead of manually setting it in `.env`.
* Add a LakeFS-backed IO manager for more complete Dagster asset versioning.
* Add model performance thresholds before allowing automatic production registration.
