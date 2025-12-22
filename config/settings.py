import os

class Config:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Ensure directories exist
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # --- Data Settings ---
    TICKER_GOLD = "geram18"
    TICKER_USD = "price_dollar_rl"
    LOOKBACK_DAYS = 30       
    TEST_SPLIT_RATIO = 0.15 
    
    # --- Feature Engineering ---
    FEATURE_COLUMNS = [
        'Gold_Log_Ret', 
        'SMA_7_Log', 'RSI_14', 'MACD_Norm', 
        'Gold_Lag_1', 'Gold_Lag_2', 'Oil_Log_Ret'
    ]
    TARGET_COLUMN = 'Target_NextDay_LogRet'

    # --- Model Hyperparameters ---
    LSTM_UNITS_1 = 64
    LSTM_UNITS_2 = 32
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.0005
    EPOCHS = 50
    BATCH_SIZE = 16
    
    # --- Trading Strategy ---
    TRANSACTION_FEE = 0.002  
    BUY_THRESHOLD = 0.0005  
    INITIAL_CAPITAL = 100_000_000

# Create a global config instance
CONF = Config()
