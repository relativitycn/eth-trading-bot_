"""
Generate synthetic BTC price data for testing when Binance API is not accessible
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_synthetic_btc_data(n_days=180, interval_minutes=1):
    """
    Generate realistic synthetic BTC price data

    Args:
        n_days: Number of days of data to generate
        interval_minutes: Interval in minutes (default 1 for 1m candles)

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Generating {n_days} days of synthetic 1m BTC data...")

    # Calculate number of candles
    n_candles = n_days * 24 * 60 // interval_minutes

    # Starting price and parameters
    base_price = 45000  # Starting BTC price
    trend = 0.00001  # Slight upward trend
    volatility = 0.0015  # 0.15% volatility per candle

    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=n_days)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=n_candles)

    # Generate price walk with realistic characteristics
    np.random.seed(42)  # For reproducibility

    # Generate returns with trend and volatility
    returns = np.random.normal(trend, volatility, n_candles)

    # Add some autocorrelation (price momentum)
    for i in range(1, len(returns)):
        returns[i] += 0.3 * returns[i-1]  # Momentum effect

    # Calculate close prices
    close_prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLV from close prices
    ohlcv_data = []

    for i in range(n_candles):
        close = close_prices[i]

        # High and low based on volatility
        high = close * (1 + abs(np.random.normal(0, volatility * 0.5)))
        low = close * (1 - abs(np.random.normal(0, volatility * 0.5)))

        # Open is somewhere between previous close and current close
        if i == 0:
            open_price = close
        else:
            open_price = close_prices[i-1] + np.random.normal(0, volatility * close_prices[i-1] * 0.3)

        # Volume (realistic with some correlation to price movement)
        base_volume = np.random.lognormal(10, 1.5)  # Lognormal distribution
        volume_multiplier = 1 + abs(returns[i]) * 10  # Higher volume on big moves
        volume = base_volume * volume_multiplier

        ohlcv_data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': max(open_price, high, close),
            'low': min(open_price, low, close),
            'close': close,
            'volume': volume,
            'close_time': timestamps[i] + timedelta(minutes=interval_minutes),
            'quote_volume': volume * close,
            'trades': int(np.random.uniform(50, 500)),
            'taker_buy_base': volume * np.random.uniform(0.4, 0.6),
            'taker_buy_quote': volume * close * np.random.uniform(0.4, 0.6),
            'ignore': 0
        })

    df = pd.DataFrame(ohlcv_data)
    df.set_index('timestamp', inplace=True)

    print(f"Generated {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df


if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)

    df = generate_synthetic_data()
    df.to_csv('data/raw_data.csv')
    print("\nSynthetic data saved to data/raw_data.csv")
    print("\nData preview:")
    print(df.head())
