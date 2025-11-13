"""
Feature Engineering Module
Creates technical indicators and features for ML model
All indicators implemented manually without external TA libraries
"""

import pandas as pd
import numpy as np
import config


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_diff = macd_line - signal_line
    return macd_line, signal_line, macd_diff


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = calculate_atr(high, low, close, period=1)
    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx


def calculate_cci(high, low, close, period=20):
    """Calculate Commodity Channel Index"""
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad)
    return cci


def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr


def calculate_mfi(high, low, close, volume, period=14):
    """Calculate Money Flow Index"""
    tp = (high + low + close) / 3
    mf = tp * volume

    mf_positive = mf.copy()
    mf_negative = mf.copy()

    mf_positive[tp <= tp.shift()] = 0
    mf_negative[tp >= tp.shift()] = 0

    mf_positive_sum = mf_positive.rolling(window=period).sum()
    mf_negative_sum = mf_negative.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + mf_positive_sum / mf_negative_sum))
    return mfi


def calculate_obv(close, volume):
    """Calculate On-Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


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
        df[f'rsi_{period}'] = calculate_rsi(df['close'], period=period)

    # MACD
    for fast, slow, signal in config.MACD_PARAMS:
        macd_line, signal_line, macd_diff = calculate_macd(df['close'], fast=fast, slow=slow, signal=signal)
        df[f'macd_{fast}_{slow}'] = macd_line
        df[f'macd_signal_{fast}_{slow}'] = signal_line
        df[f'macd_diff_{fast}_{slow}'] = macd_diff

    # Bollinger Bands
    bb_high, bb_mid, bb_low = calculate_bollinger_bands(df['close'], period=config.BOLLINGER_PERIOD)
    df['bb_high'] = bb_high
    df['bb_mid'] = bb_mid
    df['bb_low'] = bb_low
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])

    # ATR (Volatility)
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], period=config.ATR_PERIOD)
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
    df['obv'] = calculate_obv(df['close'], df['volume'])
    df['obv_change'] = df['obv'].pct_change()

    # Stochastic Oscillator
    stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d

    # ADX (Trend Strength)
    df['adx'] = calculate_adx(df['high'], df['low'], df['close'])

    # CCI (Commodity Channel Index)
    df['cci'] = calculate_cci(df['high'], df['low'], df['close'])

    # Williams %R
    df['williams_r'] = calculate_williams_r(df['high'], df['low'], df['close'])

    # Money Flow Index
    df['mfi'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'])

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
