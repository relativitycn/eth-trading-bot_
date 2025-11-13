"""
Data fetching module for Binance historical data
"""

import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
from tqdm import tqdm
import config


class BinanceDataFetcher:
    def __init__(self, api_key=None, api_secret=None):
        """
        Initialize Binance client
        For historical data, API keys are not required
        """
        self.client = Client(api_key, api_secret)

    def fetch_historical_klines(self, symbol, interval, lookback_days):
        """
        Fetch historical candlestick data from Binance

        Args:
            symbol: Trading pair (e.g., 'BTCUSDC')
            interval: Candle interval (e.g., '1m')
            lookback_days: Number of days to look back

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching {lookback_days} days of {interval} data for {symbol}...")

        # Calculate start and end times
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        # Convert to milliseconds
        start_str = str(int(start_time.timestamp() * 1000))
        end_str = str(int(end_time.timestamp() * 1000))

        # Fetch data in chunks (Binance limit is 1000 candles per request)
        all_klines = []
        current_start = start_str

        with tqdm(total=lookback_days, desc="Downloading") as pbar:
            while True:
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=current_start,
                    end_str=end_str,
                    limit=1000
                )

                if not klines:
                    break

                all_klines.extend(klines)

                # Update progress
                last_timestamp = klines[-1][0]
                days_fetched = (last_timestamp - int(start_str)) / (1000 * 60 * 60 * 24)
                pbar.update(days_fetched - pbar.n)

                # Check if we've reached the end
                if len(klines) < 1000:
                    break

                # Move to next batch
                current_start = str(klines[-1][0] + 1)

        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        print(f"Successfully fetched {len(df)} candles")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")

        return df

    def save_data(self, df, filename='data/raw_data.csv'):
        """Save dataframe to CSV"""
        df.to_csv(filename)
        print(f"Data saved to {filename}")

    def load_data(self, filename='data/raw_data.csv'):
        """Load dataframe from CSV"""
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"Data loaded from {filename}: {len(df)} candles")
        return df


if __name__ == "__main__":
    # Test the data fetcher
    import os
    os.makedirs('data', exist_ok=True)

    fetcher = BinanceDataFetcher()
    df = fetcher.fetch_historical_klines(
        symbol=config.SYMBOL,
        interval=config.INTERVAL,
        lookback_days=config.LOOKBACK_DAYS
    )

    fetcher.save_data(df)
    print("\nData preview:")
    print(df.head())
    print(f"\nShape: {df.shape}")
