# GitHub Setup Guide

## Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Create a new repository with your preferred name (e.g., `btc-ml-trading-bot`)
3. Choose Public or Private
4. **DON'T** initialize with README

## Step 2: Connect Local Code to GitHub

After creating the repository, GitHub will show you a URL like:
```
https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

Run these commands in your terminal (replace with your actual URL):

```bash
# Navigate to project directory
cd /home/user/eth-trading-bot_

# Remove current remote
git remote remove origin

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Verify it's set correctly
git remote -v

# Push your code to GitHub
git push -u origin claude/ml-trading-bot-backtest-011CV5wsvBg87iMRx52YQ4Mm

# Or if you want to push to main branch:
git checkout -b main
git merge claude/ml-trading-bot-backtest-011CV5wsvBg87iMRx52YQ4Mm
git push -u origin main
```

## Step 3: Verify on GitHub

1. Go to your repository URL on GitHub
2. You should see all your code files
3. README.md will display automatically on the main page

## Step 4: Set Up GitHub Secrets (for Binance API - Optional)

If you want to use Binance API in the future:

1. Go to your repo → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add:
   - Name: `BINANCE_API_KEY`
   - Value: Your Binance API key
4. Add another:
   - Name: `BINANCE_API_SECRET`
   - Value: Your Binance API secret

## Step 5: Running the Bot

### Local Testing (on your machine)

```bash
# Install dependencies
pip install -r requirements.txt

# Run with synthetic data
python main.py --synthetic

# Run with real Binance data (requires API access)
python main.py

# Run with grid search for better model
python main.py --grid-search
```

### GitHub Actions (Automated)

See `.github/workflows/trading-bot.yml` for automated execution setup.

## Troubleshooting

### Authentication Error
If you get authentication errors when pushing:

**Option 1: Use Personal Access Token (Recommended)**
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name, select "repo" scope
4. Copy the token
5. When git asks for password, use the token instead

**Option 2: Use SSH**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
# Copy your public key:
cat ~/.ssh/id_ed25519.pub

# Change remote to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
```

### Large Files Error
If git complains about large files, use Git LFS:
```bash
git lfs install
git lfs track "*.csv"
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

## Project Structure

```
btc-ml-trading-bot/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration parameters
├── main.py                      # Main pipeline script
├── data_fetcher.py             # Binance data fetching
├── generate_synthetic_data.py  # Synthetic data generator
├── target_engineering.py       # Target calculation
├── feature_engineering.py      # Feature creation
├── model_training.py           # ML model training
├── backtesting.py              # Backtesting engine
├── data/                       # Data files (gitignored)
├── models/                     # Trained models (gitignored)
└── outputs/                    # Results and plots (gitignored)
```

## Next Steps

1. ✅ Push code to GitHub
2. ✅ Test with synthetic data
3. ⬜ Test with real Binance data
4. ⬜ Optimize model parameters
5. ⬜ Add risk management improvements
6. ⬜ Paper trade before going live
