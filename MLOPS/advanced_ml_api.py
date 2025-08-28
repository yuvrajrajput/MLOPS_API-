# advanced_ml_api.py
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import aiohttp
from prometheus_fastapi_instrumentator import Instrumentator
import structlog
import joblib
from datetime import datetime
import numpy as np

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    id: Optional[str] = None
    features: List[float]
    model_id: str = "default"

class PredictionResponse(BaseModel):
    id: Optional[str]
    prediction: float
    confidence: float
    model_id: str
    timestamp: datetime

class ModelInfo(BaseModel):
    model_id: str
    status: str
    loaded_at: Optional[datetime]
    path: Optional[str]

# Advanced middleware setup
app = FastAPI(
    title="Advanced ML API",
    description="Production-ready ML inference service",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Structured logging
logger = structlog.get_logger()

# Dependency injection for model loading
async def get_model_manager():
    return ModelManager()

class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        # Load a default dummy model for demonstration
        self._load_default_model()
    
    def _load_default_model(self):
        """Load a simple dummy model for demonstration"""
        try:
            # Create a simple dummy model (linear regression)
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            # Train on dummy data
            X = np.random.rand(100, 5)
            y = np.random.rand(100)
            model.fit(X, y)
            
            self.models["default"] = model
            self.model_metadata["default"] = {
                "loaded_at": datetime.utcnow(),
                "path": "dummy_model",
                "status": "ready"
            }
            logger.info("Default model loaded")
        except Exception as e:
            logger.error("Default model loading failed", error=str(e))
    
    async def load_model(self, model_id: str, model_path: str):
        """Asynchronously load model"""
        try:
            # Simulate async model loading
            await asyncio.sleep(0.1)
            model = joblib.load(model_path)
            self.models[model_id] = model
            self.model_metadata[model_id] = {
                "loaded_at": datetime.utcnow(),
                "path": model_path,
                "status": "ready"
            }
            logger.info("Model loaded", model_id=model_id)
        except Exception as e:
            logger.error("Model loading failed", model_id=model_id, error=str(e))
            raise
    
    def get_model(self, model_id: str):
        """Get a loaded model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        return self.models[model_id]
    
    def list_models(self) -> List[ModelInfo]:
        """List all loaded models"""
        return [
            ModelInfo(
                model_id=model_id,
                status=metadata["status"],
                loaded_at=metadata["loaded_at"],
                path=metadata["path"]
            )
            for model_id, metadata in self.model_metadata.items()
        ]

async def predict_single(request: PredictionRequest, model_manager: ModelManager) -> Dict[str, Any]:
    """Make a single prediction"""
    try:
        model = model_manager.get_model(request.model_id)
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        # Calculate confidence (dummy calculation for demonstration)
        confidence = 0.95
        
        return {
            "id": request.id,
            "prediction": float(prediction),
            "confidence": confidence,
            "model_id": request.model_id,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error("Prediction failed", error=str(e), request_id=request.id)
        raise

async def log_batch_metrics(total_requests: int, successful_predictions: int):
    """Log batch prediction metrics"""
    logger.info(
        "Batch prediction completed",
        total_requests=total_requests,
        successful_predictions=successful_predictions,
        success_rate=successful_predictions/total_requests if total_requests > 0 else 0
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Model info endpoint
@app.get("/models", response_model=List[ModelInfo])
async def list_models(model_manager: ModelManager = Depends(get_model_manager)):
    """List all loaded models"""
    return model_manager.list_models()

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Make a single prediction"""
    try:
        result = await predict_single(request, model_manager)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict/batch")
async def batch_predict(
    requests: List[PredictionRequest],
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Handle batch predictions efficiently"""
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large")
    
    results = []
    for request in requests:
        try:
            result = await predict_single(request, model_manager)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "request_id": getattr(request, 'id', None)})
    
    # Log batch metrics in background
    background_tasks.add_task(log_batch_metrics, len(requests), len([r for r in results if 'error' not in r]))
    
    return {"predictions": results, "total": len(requests)}

# Model loading endpoint
@app.post("/models/{model_id}/load")
async def load_model(
    model_id: str,
    model_path: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Load a new model"""
    try:
        await model_manager.load_model(model_id, model_path)
        return {"message": f"Model {model_id} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)