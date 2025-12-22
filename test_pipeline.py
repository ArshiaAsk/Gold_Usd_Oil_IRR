import os
import sys

# Add project root to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.feature_eng import FeatureEngineer
from src.utils import logger

def main():
    logger.info("=== Starting Pipeline Test ===")

    # --- Step 1: Data Loading ---
    loader = DataLoader()
    
    # Try to load local data first to save time, otherwise fetch new
    df = loader.load_raw_data()
    
    if df is None:
        logger.info("Local data not found. Fetching from APIs...")
        df = loader.fetch_data()
        if df is not None:
            loader.save_raw_data(df)
    
    if df is None:
        logger.critical("❌ Data loading failed. Exiting.")
        return

    logger.info(f"Data loaded successfully. Head:\n{df.head(3)}")

    # --- Step 2: Feature Engineering ---
    engineer = FeatureEngineer()
    
    # Process the data
    processed_df = engineer.create_features(df)
    
    # Save the result
    engineer.save_processed_data(processed_df)

    # --- Step 3: Verification ---
    logger.info("=== Verification ===")
    
    # Check for crucial columns
    required_cols = ['Gold_LogRet', 'RSI_14', 'Target_Next_LogRet', 'Gold_LogRet_Lag_1']
    missing_cols = [col for col in required_cols if col not in processed_df.columns]
    
    if not missing_cols:
        logger.info("✅ All required features exist.")
    else:
        logger.error(f"❌ Missing columns: {missing_cols}")

    logger.info(f"Final Dataset Shape: {processed_df.shape}")
    logger.info("Pipeline test finished successfully.")

if __name__ == "__main__":
    main()
