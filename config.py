"""
Configuration file for ML Trading Bot
"""

# Trading Parameters
SYMBOL = "BTCUSDC"
INTERVAL = "1m"
LOOKBACK_DAYS = 180  # How many days of historical data to fetch

# Target Definition
TARGET_LOOKAHEAD = 15  # Look ahead 15 candles (15 minutes)
LONG_THRESHOLD = 0.80  # Signal must be >= 80% for long
SHORT_THRESHOLD = 0.20  # Signal must be <= 20% for short

# Position Sizing
MAX_POSITION_SIZE = 1.0  # Maximum position size (100% of capital)
MIN_POSITION_SIZE = 0.1  # Minimum position size (10% of capital)
# Position size will scale linearly between min and max based on signal strength

# Binance Futures Fees (Maker/Taker)
MAKER_FEE = 0.0002  # 0.02%
TAKER_FEE = 0.0004  # 0.04%
# We'll use taker fees for conservative estimation

# Backtesting
INITIAL_CAPITAL = 10000  # Starting capital in USDC
LEVERAGE = 1  # Start with 1x leverage, can be increased after validation
SLIPPAGE = 0.0001  # 0.01% slippage estimate

# Train/Test/OOT Split (chronological)
TRAIN_RATIO = 0.6  # 60% for training
TEST_RATIO = 0.2   # 20% for validation/testing
OOT_RATIO = 0.2    # 20% for out-of-time testing

# Model Parameters
RANDOM_STATE = 42
N_JOBS = -1  # Use all CPU cores

# LightGBM Grid Search Parameters
LGBM_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'num_leaves': [31, 63, 127],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# Feature Engineering
RSI_PERIODS = [14, 21]
MACD_PARAMS = [(12, 26, 9)]
BOLLINGER_PERIOD = 20
ATR_PERIOD = 14
VOLUME_PERIODS = [5, 10, 20]
