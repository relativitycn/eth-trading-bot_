"""
Main Pipeline Script
Runs the complete ML trading bot pipeline:
1. Data fetching
2. Target engineering
3. Feature engineering
4. Model training (with optional grid search)
5. Backtesting
"""

import os
import pandas as pd
import argparse
from datetime import datetime

from data_fetcher import BinanceDataFetcher
from target_engineering import calculate_target, create_labels, get_signal_strength
from feature_engineering import add_technical_indicators, get_feature_columns, prepare_ml_data
from model_training import ModelTrainer
from backtesting import Backtester
import config


def create_directories():
    """Create necessary directories"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)


def fetch_data(force_download=False, use_synthetic=False):
    """
    Fetch historical data from Binance or generate synthetic data

    Args:
        force_download: If True, download fresh data even if file exists
        use_synthetic: If True, use synthetic data instead of Binance

    Returns:
        DataFrame with OHLCV data
    """
    data_file = 'data/raw_data.csv'

    if os.path.exists(data_file) and not force_download:
        print(f"Loading existing data from {data_file}")
        fetcher = BinanceDataFetcher()
        df = fetcher.load_data(data_file)
    elif use_synthetic:
        print("Generating synthetic data...")
        from generate_synthetic_data import generate_synthetic_btc_data
        df = generate_synthetic_btc_data(n_days=config.LOOKBACK_DAYS, interval_minutes=1)
        df.to_csv(data_file)
        print(f"Synthetic data saved to {data_file}")
    else:
        print("Fetching fresh data from Binance...")
        try:
            fetcher = BinanceDataFetcher()
            df = fetcher.fetch_historical_klines(
                symbol=config.SYMBOL,
                interval=config.INTERVAL,
                lookback_days=config.LOOKBACK_DAYS
            )
            fetcher.save_data(df, data_file)
        except Exception as e:
            print(f"Error fetching from Binance: {e}")
            print("Falling back to synthetic data...")
            from generate_synthetic_data import generate_synthetic_btc_data
            df = generate_synthetic_btc_data(n_days=config.LOOKBACK_DAYS, interval_minutes=1)
            df.to_csv(data_file)
            print(f"Synthetic data saved to {data_file}")

    return df


def prepare_features(df):
    """
    Prepare features: target engineering + feature engineering

    Args:
        df: Raw OHLCV DataFrame

    Returns:
        DataFrame with features, targets, and labels
    """
    print("\n" + "="*60)
    print("PREPARING FEATURES")
    print("="*60)

    # Calculate target
    df = calculate_target(df, lookahead=config.TARGET_LOOKAHEAD)

    # Create labels
    df = create_labels(df, long_threshold=config.LONG_THRESHOLD, short_threshold=config.SHORT_THRESHOLD)

    # Calculate signal strength
    df = get_signal_strength(df)

    # Add technical indicators
    df = add_technical_indicators(df)

    # Clean data
    df = prepare_ml_data(df)

    # Save
    df.to_csv('data/data_with_features.csv')
    print(f"\nData with features saved to data/data_with_features.csv")

    return df


def train_model(df, use_grid_search=False):
    """
    Train ML model

    Args:
        df: DataFrame with features and targets
        use_grid_search: If True, use grid search (slower but better)

    Returns:
        Trained ModelTrainer instance
    """
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Number of features: {len(feature_cols)}")

    # Initialize trainer
    trainer = ModelTrainer()

    # Split data
    X_train, X_test, X_oot, y_train, y_test, y_oot, _, _, _ = trainer.split_data(df, feature_cols)

    # Train
    if use_grid_search:
        print("\nUsing GRID SEARCH (this will take a while...)")
        trainer.train_with_grid_search(X_train, y_train)
    else:
        print("\nUsing SIMPLE TRAINING (faster)")
        trainer.train_simple(X_train, y_train)

    # Evaluate on all datasets
    print("\n" + "-"*60)
    train_metrics = trainer.evaluate(X_train, y_train, "TRAIN")
    test_metrics = trainer.evaluate(X_test, y_test, "TEST")
    oot_metrics = trainer.evaluate(X_oot, y_oot, "OOT")

    # Feature importance
    trainer.plot_feature_importance(feature_cols, save_path='outputs/feature_importance.png')

    # Prediction plots
    trainer.plot_predictions(y_oot, oot_metrics['predictions'], "OOT", save_path='outputs/predictions.png')

    # Save model
    trainer.save_model('models/lightgbm_model.pkl')

    return trainer


def run_backtest(df, trainer):
    """
    Run backtest on OOT data

    Args:
        df: DataFrame with features and targets
        trainer: Trained ModelTrainer instance

    Returns:
        Backtest metrics
    """
    print("\n" + "="*60)
    print("RUNNING BACKTEST")
    print("="*60)

    # Get feature columns
    feature_cols = get_feature_columns(df)

    # Split data
    X_train, X_test, X_oot, y_train, y_test, y_oot, train_idx, test_idx, oot_idx = trainer.split_data(df, feature_cols)

    # Get predictions for OOT
    predictions = trainer.model.predict(X_oot)

    # Get OOT data with labels and signal strengths
    oot_data = df.loc[oot_idx]
    labels = oot_data['label']
    signal_strengths = oot_data['signal_strength']

    # Run backtest
    backtester = Backtester(
        initial_capital=config.INITIAL_CAPITAL,
        leverage=config.LEVERAGE
    )
    metrics = backtester.run_backtest(oot_data, predictions, labels, signal_strengths, dataset_name="OOT")

    # Plot results
    backtester.plot_results(save_path='outputs/backtest_results.png')

    # Export trades
    backtester.export_trades(path='outputs/trades.csv')

    return metrics, backtester


def save_summary(trainer, backtest_metrics):
    """
    Save summary report

    Args:
        trainer: Trained ModelTrainer instance
        backtest_metrics: Backtest metrics dictionary
    """
    summary_file = 'outputs/summary.txt'

    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ML TRADING BOT - SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("CONFIGURATION\n")
        f.write("-"*60 + "\n")
        f.write(f"Symbol:              {config.SYMBOL}\n")
        f.write(f"Interval:            {config.INTERVAL}\n")
        f.write(f"Lookback Days:       {config.LOOKBACK_DAYS}\n")
        f.write(f"Target Lookahead:    {config.TARGET_LOOKAHEAD} candles\n")
        f.write(f"Long Threshold:      {config.LONG_THRESHOLD}\n")
        f.write(f"Short Threshold:     {config.SHORT_THRESHOLD}\n")
        f.write(f"Initial Capital:     ${config.INITIAL_CAPITAL:,.2f}\n")
        f.write(f"Leverage:            {config.LEVERAGE}x\n")
        f.write(f"Taker Fee:           {config.TAKER_FEE*100:.2f}%\n\n")

        f.write("BACKTEST RESULTS\n")
        f.write("-"*60 + "\n")
        f.write(f"Final Capital:       ${backtest_metrics['final_capital']:,.2f}\n")
        f.write(f"Total Return:        {backtest_metrics['total_return_pct']:+.2f}%\n")
        f.write(f"Sharpe Ratio:        {backtest_metrics['sharpe_ratio']:.4f}\n")
        f.write(f"Sortino Ratio:       {backtest_metrics['sortino_ratio']:.4f}\n")
        f.write(f"Max Drawdown:        {backtest_metrics['max_drawdown_pct']:.2f}%\n")
        f.write(f"Calmar Ratio:        {backtest_metrics['calmar_ratio']:.4f}\n")
        f.write(f"Number of Trades:    {backtest_metrics['n_trades']}\n")
        f.write(f"Win Rate:            {backtest_metrics['win_rate']:.2f}%\n")
        f.write(f"Profit Factor:       {backtest_metrics['profit_factor']:.2f}\n\n")

        if hasattr(trainer, 'best_params') and trainer.best_params:
            f.write("MODEL PARAMETERS\n")
            f.write("-"*60 + "\n")
            for param, value in trainer.best_params.items():
                f.write(f"{param}: {value}\n")

    print(f"\nSummary report saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='ML Trading Bot Pipeline')
    parser.add_argument('--download', action='store_true', help='Force download fresh data')
    parser.add_argument('--grid-search', action='store_true', help='Use grid search for model training (slower)')
    parser.add_argument('--skip-download', action='store_true', help='Skip data download and use existing data')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data instead of Binance')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ML TRADING BOT - FULL PIPELINE")
    print("="*60 + "\n")

    # Create directories
    create_directories()

    # Step 1: Fetch data
    if not args.skip_download:
        df = fetch_data(force_download=args.download, use_synthetic=args.synthetic)
    else:
        print("Skipping data download, using existing data...")
        df = pd.read_csv('data/data_with_features.csv', index_col=0, parse_dates=True)

    # Step 2: Prepare features (if not skipping download)
    if not args.skip_download:
        df = prepare_features(df)

    # Step 3: Train model
    trainer = train_model(df, use_grid_search=args.grid_search)

    # Step 4: Run backtest
    backtest_metrics, backtester = run_backtest(df, trainer)

    # Step 5: Save summary
    save_summary(trainer, backtest_metrics)

    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nOutput files:")
    print("  - outputs/backtest_results.png")
    print("  - outputs/feature_importance.png")
    print("  - outputs/predictions.png")
    print("  - outputs/trades.csv")
    print("  - outputs/summary.txt")
    print("  - models/lightgbm_model.pkl")
    print("\n")


if __name__ == "__main__":
    main()
