"""
API Client Example - How to use the Gold Price Prediction API
File: src/api/client_example.py
"""
import requests
import json
import numpy as np
import pandas as pd
from typing import Dict, List


class GoldPriceAPIClient:
    """Client for Gold Price Prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict(self, features: np.ndarray, current_price: float) -> Dict:
        """
        Make a prediction
        
        Args:
            features: Feature array of shape (30, 15)
            current_price: Current gold price
            
        Returns:
            Prediction response
        """
        payload = {
            "features": features.tolist(),
            "current_price": current_price
        }
        
        response = requests.post(
            f"{self.base_url}/predict",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.text}")
    
    def predict_with_confidence(
        self, 
        features: np.ndarray, 
        current_price: float,
        n_simulations: int = 100
    ) -> Dict:
        """Predict with confidence intervals"""
        payload = {
            "features": features.tolist(),
            "current_price": current_price
        }
        
        response = requests.post(
            f"{self.base_url}/predict-with-confidence",
            json=payload,
            params={"n_simulations": n_simulations}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.text}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        response = requests.get(f"{self.base_url}/model-info")
        return response.json()


# Example usage
def example_usage():
    """Example of how to use the API client"""
    
    # Initialize client
    client = GoldPriceAPIClient("http://localhost:8000")
    
    # 1. Check health
    print("=" * 80)
    print("1. Health Check")
    print("=" * 80)
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    # 2. Get model info
    print("\n" + "=" * 80)
    print("2. Model Information")
    print("=" * 80)
    model_info = client.get_model_info()
    print(json.dumps(model_info, indent=2))
    
    # 3. Make prediction (example with random features)
    print("\n" + "=" * 80)
    print("3. Make Prediction")
    print("=" * 80)
    
    # Create example features (30 time steps, 15 features)
    # In production, these would be real market data
    features = np.random.randn(30, 15)
    current_price = 95_000_000  # 95 million Toman
    
    result = client.predict(features, current_price)
    
    print(json.dumps(result['prediction'], indent=2))
    print(f"\n📊 Current Price: {result['prediction']['current_price']:,.0f} Toman")
    print(f"📊 Predicted Price: {result['prediction']['predicted_price']:,.0f} Toman")
    print(f"📊 Change: {result['prediction']['price_change']:+,.0f} Toman ({result['prediction']['price_change_percent']:+.2f}%)")
    
    # 4. Prediction with confidence intervals
    print("\n" + "=" * 80)
    print("4. Prediction with Confidence Intervals")
    print("=" * 80)
    
    conf_result = client.predict_with_confidence(features, current_price, n_simulations=100)
    
    pred = conf_result['prediction']
    print(f"\n📊 Predicted Price: {pred['predicted_price']:,.0f} Toman")
    print(f"📊 95% CI: [{pred['confidence_interval_95']['lower']:,.0f}, {pred['confidence_interval_95']['upper']:,.0f}]")
    print(f"📊 68% CI: [{pred['confidence_interval_68']['lower']:,.0f}, {pred['confidence_interval_68']['upper']:,.0f}]")
    print(f"📊 Std Dev: {pred['std_dev']:,.0f} Toman")


if __name__ == "__main__":
    example_usage()
