"""
Target Engineering Module
Implements the custom target definition: % of next N candles above current price
"""

import pandas as pd
import numpy as np
import config


def calculate_target(df, lookahead=config.TARGET_LOOKAHEAD):
    """
    Calculate target: % of next N candles with close price > current close

    For each candle at time t with price P_t:
    - Look at next N candles
    - Count how many have close price > P_t
    - Target = count / N

    Args:
        df: DataFrame with 'close' column
        lookahead: Number of candles to look ahead

    Returns:
        DataFrame with added 'target' column
    """
    print(f"Calculating target with {lookahead}-candle lookahead...")

    df = df.copy()
    targets = []

    close_prices = df['close'].values

    for i in range(len(df)):
        # Get future candles
        future_start = i + 1
        future_end = i + 1 + lookahead

        # If not enough future data, mark as NaN
        if future_end > len(df):
            targets.append(np.nan)
            continue

        # Get future closes
        future_closes = close_prices[future_start:future_end]
        current_close = close_prices[i]

        # Count how many future candles are above current
        count_above = np.sum(future_closes > current_close)

        # Calculate percentage
        target = count_above / lookahead

        targets.append(target)

    df['target'] = targets

    # Remove rows with NaN targets (last N candles)
    df_clean = df.dropna(subset=['target'])

    print(f"Target calculated. Valid samples: {len(df_clean)}/{len(df)}")
    print(f"\nTarget distribution:")
    print(df_clean['target'].describe())

    return df_clean


def create_labels(df, long_threshold=config.LONG_THRESHOLD, short_threshold=config.SHORT_THRESHOLD):
    """
    Create trading labels based on target thresholds

    Args:
        df: DataFrame with 'target' column
        long_threshold: Target >= this → Long (1)
        short_threshold: Target <= this → Short (-1)
        Otherwise → Neutral (0)

    Returns:
        DataFrame with 'label' column
    """
    df = df.copy()

    conditions = [
        df['target'] >= long_threshold,
        df['target'] <= short_threshold,
    ]
    choices = [1, -1]  # Long, Short

    df['label'] = np.select(conditions, choices, default=0)  # Neutral

    # Statistics
    n_long = (df['label'] == 1).sum()
    n_short = (df['label'] == -1).sum()
    n_neutral = (df['label'] == 0).sum()
    total = len(df)

    print(f"\nLabel distribution:")
    print(f"  Long (1):    {n_long:6d} ({n_long/total*100:.2f}%)")
    print(f"  Short (-1):  {n_short:6d} ({n_short/total*100:.2f}%)")
    print(f"  Neutral (0): {n_neutral:6d} ({n_neutral/total*100:.2f}%)")

    return df


def get_signal_strength(df):
    """
    Calculate signal strength for position sizing

    For Long signals (target >= 0.8):
        - 0.80 → 0.0 strength (min position)
        - 1.00 → 1.0 strength (max position)

    For Short signals (target <= 0.2):
        - 0.20 → 0.0 strength (min position)
        - 0.00 → 1.0 strength (max position)

    Neutral: strength = 0
    """
    df = df.copy()

    signal_strength = np.zeros(len(df))

    # Long signals
    long_mask = df['label'] == 1
    if long_mask.any():
        # Scale from 0.8-1.0 to 0.0-1.0
        signal_strength[long_mask] = (df.loc[long_mask, 'target'] - config.LONG_THRESHOLD) / (1.0 - config.LONG_THRESHOLD)

    # Short signals
    short_mask = df['label'] == -1
    if short_mask.any():
        # Scale from 0.2-0.0 to 0.0-1.0
        signal_strength[short_mask] = (config.SHORT_THRESHOLD - df.loc[short_mask, 'target']) / config.SHORT_THRESHOLD

    df['signal_strength'] = signal_strength

    return df


if __name__ == "__main__":
    # Test target engineering
    from data_fetcher import BinanceDataFetcher

    fetcher = BinanceDataFetcher()
    df = fetcher.load_data('data/raw_data.csv')

    # Calculate targets
    df = calculate_target(df)

    # Create labels
    df = create_labels(df)

    # Calculate signal strength
    df = get_signal_strength(df)

    # Show examples
    print("\n=== LONG Signal Examples (target >= 0.8) ===")
    print(df[df['label'] == 1][['close', 'target', 'signal_strength']].head(5))

    print("\n=== SHORT Signal Examples (target <= 0.2) ===")
    print(df[df['label'] == -1][['close', 'target', 'signal_strength']].head(5))

    print("\n=== NEUTRAL Examples ===")
    print(df[df['label'] == 0][['close', 'target', 'signal_strength']].head(5))

    # Save
    df.to_csv('data/data_with_targets.csv')
    print("\nData with targets saved to data/data_with_targets.csv")
