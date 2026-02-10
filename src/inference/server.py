"""FastAPI server for fraud detection inference."""

import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predictor import FraudPredictor
from ..utils.config import InferenceConfig

# Global predictor instance
predictor: FraudPredictor | None = None


class TransactionFeatures(BaseModel):
    """Input features for a single transaction."""
    transaction_id: str | int = Field(default=0)
    amount: float = Field(..., description="Transaction amount in USD")
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: bool = False
    distance_from_home: float = Field(default=0.0, ge=0)
    is_foreign: bool = False

    # User features (optional - will use defaults if not provided)
    user_avg_amount: float = Field(default=100.0)
    user_std_amount: float = Field(default=50.0)
    user_max_amount: float = Field(default=500.0)
    user_txn_count: int = Field(default=100)
    user_foreign_count: int = Field(default=0)
    user_avg_distance: float = Field(default=10.0)
    user_max_distance: float = Field(default=50.0)
    amount_deviation: float = Field(default=0.0)
    is_high_amount: int = Field(default=0)

    # Velocity features
    txn_count_1h: int = Field(default=1)
    txn_count_6h: int = Field(default=3)
    txn_count_24h: int = Field(default=5)
    high_velocity: int = Field(default=0)

    # Merchant features
    merchant_risk_score: float = Field(default=0.01)

    # Amount features
    log_amount: float | None = None
    amount_bin: int = Field(default=3)
    amount_vs_merchant_avg: float = Field(default=1.0)

    def to_feature_dict(self) -> dict[str, float]:
        """Convert to feature dictionary for prediction."""
        import math

        data = self.model_dump()
        data.pop("transaction_id")

        # Compute log_amount if not provided
        if data["log_amount"] is None:
            data["log_amount"] = math.log1p(data["amount"])

        # Convert booleans to int
        data["is_weekend"] = int(data["is_weekend"])
        data["is_foreign"] = int(data["is_foreign"])

        return data


class BatchRequest(BaseModel):
    """Batch prediction request."""
    transactions: list[TransactionFeatures]


class PredictionResponse(BaseModel):
    """Single prediction response."""
    transaction_id: str | int
    fraud_probability: float
    is_fraud: bool
    latency_ms: float


class BatchResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[PredictionResponse]
    total_latency_ms: float
    avg_latency_ms: float
    throughput_per_sec: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    threshold: float


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model on startup."""
    global predictor

    config = InferenceConfig()
    predictor = FraudPredictor(config)

    model_path = config.model_path
    if model_path.exists():
        predictor.load_model(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        print("Start the server after training a model, or provide MODEL_PATH env var")

    yield

    # Cleanup
    predictor = None


app = FastAPI(
    title="Fraud Detection API",
    description="GPU-accelerated fraud detection using XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None and predictor.model is not None,
        threshold=predictor.threshold if predictor else 0.5,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(transaction: TransactionFeatures) -> PredictionResponse:
    """Predict fraud probability for a single transaction."""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = transaction.to_feature_dict()
    result = predictor.predict_single(features, transaction.transaction_id)

    return PredictionResponse(
        transaction_id=result.transaction_id,
        fraud_probability=result.fraud_probability,
        is_fraud=result.is_fraud,
        latency_ms=result.latency_ms,
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest) -> BatchResponse:
    """Predict fraud probabilities for a batch of transactions."""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.transactions) == 0:
        raise HTTPException(status_code=400, detail="Empty batch")

    if len(request.transactions) > 10000:
        raise HTTPException(status_code=400, detail="Batch too large (max 10000)")

    features = [t.to_feature_dict() for t in request.transactions]
    transaction_ids = [t.transaction_id for t in request.transactions]

    result = predictor.predict_batch(features, transaction_ids)

    return BatchResponse(
        predictions=[
            PredictionResponse(
                transaction_id=p.transaction_id,
                fraud_probability=p.fraud_probability,
                is_fraud=p.is_fraud,
                latency_ms=p.latency_ms,
            )
            for p in result.predictions
        ],
        total_latency_ms=result.total_latency_ms,
        avg_latency_ms=result.avg_latency_ms,
        throughput_per_sec=result.throughput_per_sec,
    )


@app.post("/threshold")
async def set_threshold(threshold: float) -> dict:
    """Update the fraud decision threshold."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")

    predictor.set_threshold(threshold)
    return {"threshold": threshold, "status": "updated"}
