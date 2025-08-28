# ml_service.py - Core ML service structure
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from typing import List, Dict, Any
import asyncio
import aioredis

app = FastAPI(title="ML Inference Service", version="1.0.0")

# Configuration
class Config:
    MODEL_PATH = "models/classifier.pkl"
    REDIS_URL = "redis://localhost:6379"
    LOG_LEVEL = "INFO"

# Request/Response Models
class PredictionRequest(BaseModel):
    features: List[float]
    model_version: str = "v1.0"
    
class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    processing_time_ms: float

# Global model storage
models = {}
redis_client = None

@app.on_event("startup")
async def startup_event():
    global redis_client
    # Load models
    try:
        models["v1.0"] = joblib.load(Config.MODEL_PATH)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise
    
    # Initialize Redis for caching
    redis_client = aioredis.from_url(Config.REDIS_URL)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    # Validate input
    if len(request.features) != 10:  # Assuming 10 features
        raise HTTPException(status_code=400, detail="Expected 10 features")
    
    # Check cache first
    cache_key = f"prediction:{hash(tuple(request.features))}:{request.model_version}"
    cached_result = await redis_client.get(cache_key)
    
    if cached_result:
        return PredictionResponse.parse_raw(cached_result)
    
    # Get model
    model = models.get(request.model_version)
    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")
    
    # Make prediction
    features_array = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    confidence = max(model.predict_proba(features_array)[0])
    
    processing_time = (time.time() - start_time) * 1000
    
    response = PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        model_version=request.model_version,
        processing_time_ms=processing_time
    )
    
    # Cache result for 1 hour
    await redis_client.setex(cache_key, 3600, response.json())
    
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": list(models.keys())}