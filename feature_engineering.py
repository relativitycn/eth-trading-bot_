"""
Feature Engineering Module
Creates technical indicators and features for ML model
"""

import pandas as pd
import numpy as np
import ta
import config


def add_technical_indicators(df):
    """
    Add comprehensive technical indicators as features

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added feature columns
    """
    print("Engineering features...")
    df = df.copy()

    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Price momentum
    for period in [5, 10, 15, 30, 60]:
        df[f'price_change_{period}'] = df['close'].pct_change(period)
        df[f'high_low_ratio_{period}'] = (df['high'].rolling(period).max() - df['low'].rolling(period).min()) / df['close']

    # RSI
    for period in config.RSI_PERIODS:
        df[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()

    # MACD
    for fast, slow, signal in config.MACD_PARAMS:
        macd = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        df[f'macd_{fast}_{slow}'] = macd.macd()
        df[f'macd_signal_{fast}_{slow}'] = macd.macd_signal()
        df[f'macd_diff_{fast}_{slow}'] = macd.macd_diff()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=config.BOLLINGER_PERIOD)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])

    # ATR (Volatility)
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=config.ATR_PERIOD).average_true_range()
    df['atr_percent'] = df['atr'] / df['close']

    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

    # Volume features
    df['volume_change'] = df['volume'].pct_change()

    for period in config.VOLUME_PERIODS:
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']

    # OBV (On-Balance Volume)
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['obv_change'] = df['obv'].pct_change()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # ADX (Trend Strength)
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()

    # CCI (Commodity Channel Index)
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()

    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()

    # Money Flow Index
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()

    # Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day

    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
        df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
        df[f'returns_kurt_{window}'] = df['returns'].rolling(window).kurt()

    print(f"Features engineered. Total columns: {len(df.columns)}")

    return df


def get_feature_columns(df):
    """
    Get list of feature columns (exclude target, label, and OHLCV)

    Args:
        df: DataFrame with all columns

    Returns:
        List of feature column names
    """
    exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore',
                    'target', 'label', 'signal_strength']

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols


def prepare_ml_data(df):
    """
    Prepare data for ML by removing NaN and infinite values

    Args:
        df: DataFrame with features and targets

    Returns:
        Clean DataFrame ready for ML
    """
    print("\nPreparing data for ML...")
    print(f"Initial shape: {df.shape}")

    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN values
    df_clean = df.dropna()

    print(f"After cleaning: {df_clean.shape}")
    print(f"Rows removed: {len(df) - len(df_clean)}")

    return df_clean


if __name__ == "__main__":
    # Test feature engineering
    import pandas as pd

    df = pd.read_csv('data/data_with_targets.csv', index_col=0, parse_dates=True)

    # Add features
    df = add_technical_indicators(df)

    # Clean data
    df = prepare_ml_data(df)

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\nNumber of features: {len(feature_cols)}")
    print("\nFeature columns:")
    for col in feature_cols[:20]:  # Show first 20
        print(f"  - {col}")
    print(f"  ... and {len(feature_cols) - 20} more")

    # Save
    df.to_csv('data/data_with_features.csv')
    print(f"\nData with features saved to data/data_with_features.csv")
