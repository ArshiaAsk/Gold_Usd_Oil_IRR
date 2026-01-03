"""
FastAPI Application for Gold Price Prediction
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

from predictor import GoldPricePredictor, FeatureBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Gold Price Prediction API",
    description="LSTM-based Gold Price Prediction Service for Iranian Market",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Global predictor instance (loaded once at startup)
predictor: Optional[GoldPricePredictor]
feature_builder: Optional[FeatureBuilder]



# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for price prediction"""
    features: List[List[float]] = Field(
        ...,
        description="Feature array of shape (sequence_length, n_features)",
        min_items=30,
        max_items=30
    )
    current_price: float = Field(
        ...,
        description="Current gold price in Toman",
        gt=0
    )

    @validator('features')
    def validate_features(cls, v):
        if not all(len(row) == 15 for row in v):
            raise ValueError("Each feature row must have exactly 15 features")
        return v



class HistoricalDataRequest(BaseModel):
    """Request model using historical data"""
    historical_data: List[Dict[str, float]] = Field(
        ...,
        description="Historical market data (at least 30 days)",
        min_items=30
    )
    current_price: float = Field(..., gt=0)



class PredictionResponse(BaseModel):
    """Response model for prediction"""
    success: bool
    prediction: Dict[str, float]
    timestamp: str



class HealthResponse(BaseModel):
    """Health ckeck response"""
    status: bool
    model_loaded: bool
    timestamp: str



class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    detail: Optional[str] = None



# Stratup event - Load model once 
@app.on_event("startup")
async def load_model():
    """Load model and scalers on startup"""
    global predictor, feature_builder
    
    try:
        # Define paths (adjust based on your structure)
        base_path = Path(__file__).parent.parent.parent
        model_path = base_path / "models" / "gold_lstm_v2.keras"
        scaler_X_path = base_path / "models" / "scaler_X.pkl"
        scaler_y_path = base_path / "models" / "scaler_y.pkl"

        # Load predictor
        predictor = GoldPricePredictor(
            model_path=str(model_path),
            scaler_X_path=str(scaler_X_path),
            scaler_y_path=str(scaler_y_path)
        )
        
        # Initialize feature builder
        feature_builder = FeatureBuilder(sequence_length=30)
        
        logger.info("✅ Model and predictor loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise



# API Endpoint
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Gold Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }



@app.get("/health", response_model=HealthResponse)
async def health_check():
    "Health check endpoint"
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        timestamp=datetime.now().isoformat()
    )



@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """  
    Predict next day gold price
    
    - **features**: 2D array of shape (30, 15) with feature values
    - **current_price**: Current gold price in Toman
    
    Returns prediction with price change and percentage
    """
    if predictor in None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert to numpy array
        features = np.array(request.features)

        # Make prediction
        result = predictor.predict_price(features, request.current_price)

        return PredictionResponse(
            success=True,
            prediction=result,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction Failed: {str(e)}"
        )
    


@app.post("/predict_with_confidence", response_model=PredictionResponse)
async def predict_with_confidence(
    request: PredictionRequest,
    n_simulations: int = 100
):
    """
    Predict with Monte Carlo confidence intervals
    
    - **features**: 2D array of shape (30, 15)
    - **current_price**: Current price
    - **n_simulations**: Number of MC simulations (default: 100)
    
    Returns prediction with 68% and 95% confidence intervals
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        features = np.array(request.features)

        result = predictor.predict_with_confidence(
            features,
            request.current_price,
            n_simulations=n_simulations
        )

        return PredictionResponse(
            success=True,
            prediction=result,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )



@app.post("/predict-from-history", response_model=PredictionResponse)
async def predict_from_history(request: HistoricalDataRequest):
    """
    Predict using historical market data
    
    Automatically builds features from historical data
    
    - **historical_data**: List of dicts with market data (at least 30 days)
    - **current_price**: Current gold price
    
    Required fields in historical_data:
    - Gold_LogRet, USD_LogRet, Ounce_LogRet, Oil_LogRet
    - SMA_7, RSI_14, MACD, MACD_Signal
    - Bollinger_Upper, Bollinger_Lower
    - Gold_LogRet_Lag_1, Gold_LogRet_Lag_2, Gold_LogRet_Lag_3
    - USD_LogRet_Lag_1, USD_LogRet_Lag_2
    """
    if predictor is None or feature_builder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.historical_data)

        # Build features
        features = feature_builder.build_features_from_history(df)

        # Make Prediction
        result = predictor.predict_price(features, request.current_price)

        return PredictionResponse(
            success=True,
            prediction=result,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )



@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try: 
        model_summary = {
            "model_path": predictor.model_path,
            "input_shape": str(predictor.model.input_shape),
            "output_shape": str(predictor.model.output_shape),
            "total_parameters": int(predictor.model.count_params()),
            "expected_features": {
                "sequence_length": 30,
                "n_features": 15,
                "features_name": [
                    'Gold_LogRet', 'USD_LogRet', 'Ounce_LogRet', 'Oil_LogRet',
                    'SMA_7', 'RSI_14', 'MACD', 'MACD_Signal',
                    'Bollinger_Upper', 'Bollinger_Lower',
                    'Gold_LogRet_Lag_1', 'Gold_LogRet_Lag_2', 'Gold_LogRet_Lag_3',
                    'USD_LogRet_Lag_1', 'USD_LogRet_Lag_2'
                ]
            }
        }

        return model_summary
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {e}"
        )
    


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return ErrorResponse(
        error="Invalid input",
        detail=str(exc)
    )



@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return ErrorResponse(
        error="Internal server error",
        detail=str(exc)
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)