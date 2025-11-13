# Quick Start Guide

## 🚀 Run Locally in 5 Minutes

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Your First Backtest
```bash
# Test with synthetic data (no API needed)
python main.py --synthetic
```

That's it! The bot will:
- ✅ Generate 180 days of synthetic BTC data
- ✅ Engineer 94 features
- ✅ Train a LightGBM model
- ✅ Run backtest
- ✅ Generate visualizations in `outputs/` folder

## 📊 View Results

Open these files in `outputs/`:
- `backtest_results.png` - 6 comprehensive charts
- `feature_importance.png` - Top ML features
- `predictions.png` - Model accuracy
- `trades.csv` - All trade details
- `summary.txt` - Performance metrics

## 🔧 Advanced Usage

### Use Real Binance Data
```bash
# Requires internet connection to Binance API
python main.py
```

### Optimize Model (Slower but Better)
```bash
python main.py --synthetic --grid-search
```

### Use Existing Data
```bash
# Skip download, use previously fetched data
python main.py --skip-download
```

## ⚙️ Customize Parameters

Edit `config.py`:

```python
# Trading Parameters
LOOKBACK_DAYS = 180          # Data history
TARGET_LOOKAHEAD = 15        # Look 15 min ahead
LONG_THRESHOLD = 0.80        # 80% for long
SHORT_THRESHOLD = 0.20       # 20% for short

# Position Sizing
MAX_POSITION_SIZE = 1.0      # 100% of capital
MIN_POSITION_SIZE = 0.1      # 10% of capital
LEVERAGE = 1                 # Start conservative

# Backtesting
INITIAL_CAPITAL = 10000      # Starting capital
TAKER_FEE = 0.0004          # Binance futures fee
```

## 🎯 Understanding the Strategy

### Target Definition (Your Innovation!)
For each 1-minute candle:
1. Look at next 15 candles (15 minutes)
2. Count how many close above current price
3. Calculate percentage

**Example:**
- Current: $45,000
- Next 15 candles: 13 are above $45,000
- Target = 13/15 = 86.7%
- **Signal: LONG** (above 80% threshold)

### Position Sizing
Signal strength scales position:
- Weak signal (just at 80%) → 10% position
- Medium signal (90%) → 55% position
- Strong signal (100%) → 100% position

### Exit Strategy
Currently: Hold for exactly 15 candles (15 minutes)

## 🐛 Troubleshooting

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### Binance API Errors
Use synthetic data for testing:
```bash
python main.py --synthetic
```

### Memory Errors
Reduce data in `config.py`:
```python
LOOKBACK_DAYS = 90  # Instead of 180
```

## 📈 Next Steps for Profit

The synthetic data test shows the system works. To make it profitable:

1. **Test with Real Data**
   ```bash
   python main.py  # Uses Binance API
   ```

2. **Tune Parameters**
   - Try different thresholds (75/25, 70/30)
   - Test various lookahead periods (10, 20, 30 min)

3. **Add Risk Management**
   - Stop losses
   - Take profits
   - Max drawdown limits

4. **Paper Trade**
   - Test live without real money
   - Monitor for 2-4 weeks

5. **Go Live (Small)**
   - Start with minimum capital
   - Monitor closely
   - Scale gradually

## 📚 Learn More

- **Full Documentation**: See README.md
- **GitHub Setup**: See GITHUB_SETUP.md
- **Configuration**: See config.py
- **Code Structure**: See docstrings in each .py file

## ⚠️ Disclaimer

This is for educational purposes. Trading involves risk. Never trade with money you can't afford to lose. Past performance doesn't guarantee future results.

## 💬 Support

Open an issue on GitHub if you encounter problems or have questions!
