from typing import Any

import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/BikeRentalDemandModel@production",
)

# define JSON format
class BikeRentalFeatures(BaseModel):
    temperature_c: float
    humidity: float
    windspeed_kmh: float
    is_weekend: int
    is_holiday: int
    is_workday: int
    rush_hour: int
    was_missing_hour: int
    hour_sin: float
    hour_cos: float
    weekday_sin: float
    weekday_cos: float
    month_sin: float
    month_cos: float
    total_lag_24h: float
    total_lag_7d: float
    total_24h_mean: float
    total_7d_mean: float
    same_hour_weekday_4w_mean: float
    conditions: str
    season: str

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


@app.get("/model-info", tags=["Model"])
def model_info() -> dict[str, str]:
    return {
        "model_uri": MODEL_URI,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_status": "loaded",
    }


# check api server status
@app.get("/health", tags=["General"])
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: BikeRentalFeatures) -> PredictionResponse:
    input_data = pd.DataFrame([features.model_dump()])

    try:
        prediction = app.state.model.predict(input_data)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(error)}",
        ) from error

    return PredictionResponse(
        prediction=float(prediction[0]),
        model_uri=MODEL_URI,
    )

# add root page
@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    return {
        "message": "Bike Rental Demand Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


# api improvement
class BatchPredictionRequest(BaseModel):
    records: list[BikeRentalFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    model_uri: str

@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    input_data = pd.DataFrame(
        [record.model_dump() for record in request.records]
    )

    try:
        predictions = app.state.model.predict(input_data)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(error)}",
        ) from error

    return BatchPredictionResponse(
        predictions=[float(prediction) for prediction in predictions],
        model_uri=MODEL_URI,
    )