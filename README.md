# ML-Based BTC Trading Bot

A machine learning-based trading bot for BTC/USDC futures on Binance, designed to maximize Sharpe ratio and leverage effectiveness.

## Overview

This trading bot uses a novel target definition based on future price movements and trains a LightGBM model to predict optimal long/short signals. The system includes comprehensive backtesting with realistic fees and position sizing based on signal strength.

## Key Features

- **Custom Target Definition**: Instead of simple price prediction, targets are defined as the percentage of next N candles with prices above the current level
- **Signal Thresholds**:
  - Long signal: ≥80% of next 15 candles above current price
  - Short signal: ≤20% of next 15 candles above current price
- **Position Sizing**: Scales with signal strength (0-100%)
- **LightGBM Model**: Fast, efficient gradient boosting with optional grid search
- **Comprehensive Features**: 100+ technical indicators including RSI, MACD, Bollinger Bands, volume indicators, etc.
- **Realistic Backtesting**: Includes fees (0.04%), slippage, and proper position management
- **Performance Metrics**: Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor, and more

## Project Structure

```
.
├── config.py                 # Configuration parameters
├── data_fetcher.py          # Binance data fetching
├── target_engineering.py    # Custom target calculation
├── feature_engineering.py   # Technical indicator features
├── model_training.py        # LightGBM training with grid search
├── backtesting.py           # Backtesting engine
├── main.py                  # Main pipeline script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Simple Training)

Run the full pipeline with default parameters:

```bash
python main.py
```

This will:
1. Download 180 days of 1-minute BTC/USDC data from Binance
2. Calculate custom targets and labels
3. Engineer 100+ technical features
4. Train a LightGBM model
5. Run backtest on out-of-time (OOT) data
6. Generate performance reports and visualizations

### Advanced Usage

**Force download fresh data:**
```bash
python main.py --download
```

**Use grid search for better model (slower):**
```bash
python main.py --grid-search
```

**Use existing data (skip download):**
```bash
python main.py --skip-download
```

### Running Individual Modules

**Fetch data only:**
```bash
python data_fetcher.py
```

**Test target engineering:**
```bash
python target_engineering.py
```

**Test feature engineering:**
```bash
python feature_engineering.py
```

**Train model only:**
```bash
python model_training.py
```

**Run backtest only:**
```bash
python backtesting.py
```

## Configuration

Edit `config.py` to customize:

- **Trading Parameters**: Symbol, interval, lookback period
- **Target Definition**: Lookahead period, long/short thresholds
- **Position Sizing**: Min/max position size, leverage
- **Fees & Slippage**: Binance futures fees, slippage estimates
- **Model Parameters**: LightGBM hyperparameters, grid search space
- **Feature Engineering**: Indicator periods and parameters

## Target Definition Explained

The core innovation is the target definition:

For each candle at time `t` with price `P_t`:
1. Look at the next 15 candles (15 minutes ahead)
2. Count how many have close prices > `P_t`
3. Target = count / 15 (as a percentage)

**Examples:**
- Target = 100% → All future candles are higher → **Strong Long**
- Target = 80% → 12/15 future candles higher → **Long Signal**
- Target = 50% → Half higher, half lower → **Neutral**
- Target = 20% → 3/15 future candles higher → **Short Signal**
- Target = 0% → All future candles lower → **Strong Short**

This captures both direction and conviction of price movement.

## Output Files

After running the pipeline, check the `outputs/` directory:

- `backtest_results.png` - Comprehensive backtest visualizations
- `feature_importance.png` - Top features by importance
- `predictions.png` - Model prediction analysis
- `trades.csv` - Detailed trade log
- `summary.txt` - Text summary of results

The trained model is saved to `models/lightgbm_model.pkl`.

## Performance Metrics

The backtester calculates:

- **Total Return %** - Overall profit/loss
- **Sharpe Ratio** - Risk-adjusted returns (annualized)
- **Sortino Ratio** - Downside risk-adjusted returns
- **Max Drawdown %** - Largest peak-to-trough decline
- **Calmar Ratio** - Return / Max Drawdown
- **Win Rate %** - Percentage of profitable trades
- **Profit Factor** - Gross profit / Gross loss
- **Average Win/Loss** - Mean PnL of winning/losing trades

## Data Requirements

- Minimum: 60 days of historical data
- Recommended: 180+ days for better model training
- Data is fetched from Binance public API (no API keys required for historical data)

## Trading Logic

1. **Feature Calculation**: Calculate 100+ technical indicators for current candle
2. **Model Prediction**: LightGBM predicts target probability (0-1)
3. **Signal Generation**:
   - If prediction ≥ 0.8 → Long signal
   - If prediction ≤ 0.2 → Short signal
   - Otherwise → No trade
4. **Position Sizing**: Scale position based on signal strength
5. **Entry**: Open position at current price
6. **Exit**: Close after 15 candles (15 minutes)

## Risk Management

- **Maximum Position Size**: 100% of capital (configurable)
- **Leverage**: Default 1x (can be increased after validation)
- **Holding Period**: Fixed at 15 candles
- **Fees**: 0.04% taker fee per trade (entry + exit = 0.08% total)
- **Slippage**: 0.01% per trade

## Future Improvements

Potential enhancements:
- Dynamic stop-loss and take-profit
- Multiple timeframe analysis
- Ensemble models (XGBoost, Neural Networks)
- Walk-forward optimization
- Live trading integration
- Risk management based on volatility
- Market regime detection

## Disclaimer

This is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and never trade with money you cannot afford to lose.

## License

MIT License - Feel free to use and modify for your own purposes.

## Contributing

Suggestions and improvements are welcome! Please open an issue or submit a pull request.
