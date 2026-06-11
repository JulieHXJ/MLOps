from typing import Any
import math
import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from dg_pipeline.utils.data_engineering import create_calendar_features_from_timestamp, create_historical_demand_features_from_timestamp
from dg_pipeline.utils.feature_config import FEATURE_SETS


load_dotenv()

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/BikeRentalDemandModel@production",
)

HISTORICAL_DEMAND_PATH = os.getenv(
    "HISTORICAL_DEMAND_PATH",
    "data/processed/historical_demand.csv",
)

# define JSON format
class RawBikeRentalInput(BaseModel):
    timestamp: str = Field(..., example="2012-06-01T08:00:00")
    temperature_c: float = Field(..., example=18.5)
    humidity: float = Field(..., example=65.0)
    windspeed_kmh: float = Field(..., example=12.0)
    conditions: str = Field(..., example="Clear")
    is_holiday: int = Field(..., example=0)


API_MODEL_FEATURES = [
    "temperature_c",
    "humidity",
    "windspeed_kmh",
    "is_weekend",
    "is_holiday",
    "is_workday",
    "rush_hour",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "conditions",
    "season",
]

# raw feature helper
def build_model_features_from_raw_input(
    raw_input: RawBikeRentalInput,
    historical_demand: pd.DataFrame,
) -> pd.DataFrame:
    timestamp = pd.to_datetime(raw_input.timestamp)

    calendar_features = create_calendar_features_from_timestamp(
        timestamp=timestamp,
        is_holiday=raw_input.is_holiday,
    )

    historical_features = create_historical_demand_features_from_timestamp(
        timestamp=timestamp,
        historical_demand=historical_demand,
    )

    features = {
        "temperature_c": raw_input.temperature_c,
        "humidity": raw_input.humidity,
        "windspeed_kmh": raw_input.windspeed_kmh,
        **calendar_features,
        "was_missing_hour": 0,
        **historical_features,
        "conditions": raw_input.conditions,
    }

    input_data = pd.DataFrame([features])

    engineered_features = FEATURE_SETS["engineered"]["features"]

    missing_features = set(engineered_features) - set(input_data.columns)

    if missing_features:
        raise ValueError(
            f"Missing engineered model features: {missing_features}"
        )

    return input_data[engineered_features]



# output format
class PredictionResponse(BaseModel):
    prediction: float
    model_uri: str


app = FastAPI(
    title="Bike Rental Demand Prediction API",
    description="FastAPI service for predicting hourly bike rental demand using the MLflow production model.",
    version="0.1.0",
)

@app.on_event("startup")
def load_model() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)

    historical_demand = pd.read_csv(HISTORICAL_DEMAND_PATH)
    historical_demand["hour"] = pd.to_datetime(historical_demand["hour"])
    historical_demand = historical_demand.sort_values("hour").reset_index(drop=True)

    required_columns = {"hour", "total_count"}
    missing_columns = required_columns - set(historical_demand.columns)

    if missing_columns:
        raise ValueError(
            f"Historical demand data is missing columns: {missing_columns}"
        )

    app.state.historical_demand = historical_demand



@app.get("/model-info", tags=["Model"])
def model_info() -> dict[str, str]:
    model_status = "loaded" if hasattr(app.state, "model") else "not_loaded"
    historical_data_status = (
        "loaded" if hasattr(app.state, "historical_demand") else "not_loaded"
    )

    return {
        "model_uri": MODEL_URI,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_status": model_status,
        "historical_demand_path": HISTORICAL_DEMAND_PATH,
        "historical_data_status": historical_data_status,
    }


# check api server status
@app.get("/health", tags=["General"])
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
    }

# add root page
@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    return {
        "message": "Bike Rental Demand Prediction API",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "predict": "/predict",
        "predict_batch": "/predict-batch",
    }



@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: RawBikeRentalInput) -> PredictionResponse:

    if not hasattr(app.state, "model"):
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check MLflow tracking URI and model URI.",
        )
    if not hasattr(app.state, "historical_demand"):
        raise HTTPException(
            status_code=503,
            detail="Historical demand data is not loaded.",
        )

    
    try:
        input_data = build_model_features_from_raw_input(
            raw_input=features,
            historical_demand=app.state.historical_demand,
        )

        prediction = app.state.model.predict(input_data)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(error)}",
        ) from error

    return PredictionResponse(
        prediction=float(prediction[0]),
        model_uri=MODEL_URI,
    )



# api improvement
class BatchPredictionRequest(BaseModel):
    records: list[RawBikeRentalInput]


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    model_uri: str

@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    if not hasattr(app.state, "model"):
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check MLflow tracking URI and model URI.",
        )

    if not hasattr(app.state, "historical_demand"):
        raise HTTPException(
            status_code=503,
            detail="Historical demand data is not loaded.",
        )

    try:
        input_data = pd.concat(
            [
                build_model_features_from_raw_input(
                    raw_input=record,
                    historical_demand=app.state.historical_demand,
                )
                for record in request.records
            ],
            ignore_index=True,
        )

        predictions = app.state.model.predict(input_data)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(error)}",
        ) from error

    return BatchPredictionResponse(
        predictions=[float(prediction) for prediction in predictions],
        model_uri=MODEL_URI,
    )