"""
Backtesting Engine
Simulates trading with realistic fees, slippage, and position sizing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
import os


class Backtester:
    def __init__(self, initial_capital=config.INITIAL_CAPITAL, leverage=config.LEVERAGE):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.trades = []
        self.equity_curve = []

    def calculate_position_size(self, signal_strength, capital):
        """
        Calculate position size based on signal strength

        Args:
            signal_strength: 0 to 1, where 1 is strongest signal
            capital: Available capital

        Returns:
            Position size in USDC
        """
        # Scale between min and max position size
        position_fraction = config.MIN_POSITION_SIZE + \
                           (config.MAX_POSITION_SIZE - config.MIN_POSITION_SIZE) * signal_strength

        position_size = capital * position_fraction * self.leverage

        return position_size

    def run_backtest(self, df, predictions, labels, signal_strengths, dataset_name="OOT"):
        """
        Run backtest simulation

        Args:
            df: DataFrame with OHLCV data
            predictions: Model predictions (target probabilities)
            labels: Trading labels (1=long, -1=short, 0=neutral)
            signal_strengths: Signal strength for position sizing (0-1)
            dataset_name: Name of dataset for reporting

        Returns:
            Dictionary with backtest results
        """
        print(f"\n{'='*60}")
        print(f"BACKTESTING ON {dataset_name} DATA")
        print(f"{'='*60}")

        capital = self.initial_capital
        position = 0  # Current position: positive for long, negative for short
        entry_price = 0
        position_size = 0

        equity = [capital]
        timestamps = [df.index[0]]

        self.trades = []

        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_label = labels.iloc[i]
            current_strength = signal_strengths.iloc[i]
            current_time = df.index[i]

            # Close existing position first
            if position != 0:
                # Exit after holding for TARGET_LOOKAHEAD candles
                holding_period = i - entry_index

                if holding_period >= config.TARGET_LOOKAHEAD:
                    # Close position
                    if position > 0:  # Close long
                        pnl_pct = (current_price / entry_price - 1) * self.leverage
                    else:  # Close short
                        pnl_pct = (entry_price / current_price - 1) * self.leverage

                    # Subtract fees and slippage
                    total_cost = config.TAKER_FEE * 2 + config.SLIPPAGE  # Entry + exit
                    pnl_pct -= total_cost

                    # Calculate PnL
                    pnl = position_size * pnl_pct
                    capital += pnl

                    # Record trade
                    self.trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': 'long' if position > 0 else 'short',
                        'position_size': position_size,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'signal_strength': entry_strength,
                        'holding_period': holding_period
                    })

                    # Reset position
                    position = 0
                    position_size = 0

            # Open new position if no current position
            if position == 0 and current_label != 0:
                position_size = self.calculate_position_size(current_strength, capital)

                # Only trade if we have capital
                if capital > 0 and position_size > 0:
                    position = current_label  # 1 for long, -1 for short
                    entry_price = current_price
                    entry_time = current_time
                    entry_index = i
                    entry_strength = current_strength

            # Record equity
            equity.append(capital)
            timestamps.append(current_time)

        # Close any remaining position at the end
        if position != 0:
            if position > 0:
                pnl_pct = (current_price / entry_price - 1) * self.leverage
            else:
                pnl_pct = (entry_price / current_price - 1) * self.leverage

            total_cost = config.TAKER_FEE * 2 + config.SLIPPAGE
            pnl_pct -= total_cost
            pnl = position_size * pnl_pct
            capital += pnl

            self.trades.append({
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_price': entry_price,
                'exit_price': current_price,
                'position': 'long' if position > 0 else 'short',
                'position_size': position_size,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'signal_strength': entry_strength,
                'holding_period': len(df) - entry_index
            })

        # Create equity curve DataFrame
        self.equity_curve = pd.DataFrame({
            'timestamp': timestamps,
            'equity': equity
        }).set_index('timestamp')

        # Calculate metrics
        metrics = self.calculate_metrics()

        return metrics

    def calculate_metrics(self):
        """
        Calculate comprehensive performance metrics

        Returns:
            Dictionary of metrics
        """
        if not self.trades:
            print("No trades executed!")
            return {}

        trades_df = pd.DataFrame(self.trades)
        equity = self.equity_curve['equity'].values

        # Basic metrics
        total_return = (equity[-1] / self.initial_capital - 1) * 100
        n_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]

        win_rate = len(winning_trades) / n_trades * 100 if n_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else np.inf

        # Returns
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) > 0:
            avg_return = np.mean(returns)
            std_return = np.std(returns)

            # Sharpe Ratio (annualized)
            # For 1-minute data: 525600 minutes per year
            sharpe_ratio = (avg_return / std_return) * np.sqrt(525600) if std_return > 0 else 0

            # Sortino Ratio
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino_ratio = (avg_return / downside_std) * np.sqrt(525600) if downside_std > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        # Max Drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax * 100
        max_drawdown = np.min(drawdown)

        # Calmar Ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        metrics = {
            'total_return_pct': total_return,
            'final_capital': equity[-1],
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'avg_trade_pnl': trades_df['pnl'].mean(),
            'total_pnl': trades_df['pnl'].sum()
        }

        self.print_metrics(metrics)

        return metrics

    def print_metrics(self, metrics):
        """Print metrics in a nice format"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"\n💰 FINANCIAL METRICS")
        print(f"  Initial Capital:     ${self.initial_capital:,.2f}")
        print(f"  Final Capital:       ${metrics['final_capital']:,.2f}")
        print(f"  Total Return:        {metrics['total_return_pct']:+.2f}%")
        print(f"  Total PnL:           ${metrics['total_pnl']:+,.2f}")

        print(f"\n📊 RISK METRICS")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.4f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:.4f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:.4f}")

        print(f"\n📈 TRADING METRICS")
        print(f"  Number of Trades:    {metrics['n_trades']}")
        print(f"  Win Rate:            {metrics['win_rate']:.2f}%")
        print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
        print(f"  Avg Win:             ${metrics['avg_win']:+,.2f}")
        print(f"  Avg Loss:            ${metrics['avg_loss']:+,.2f}")
        print(f"  Avg Trade PnL:       ${metrics['avg_trade_pnl']:+,.2f}")

        print("="*60 + "\n")

    def plot_results(self, save_path='outputs/backtest_results.png'):
        """
        Plot backtest results

        Args:
            save_path: Path to save plot
        """
        if not self.trades or len(self.equity_curve) == 0:
            print("No trades to plot!")
            return

        trades_df = pd.DataFrame(self.trades)

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        # Equity curve
        axes[0, 0].plot(self.equity_curve.index, self.equity_curve['equity'], linewidth=1.5)
        axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0, 0].set_title('Equity Curve', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Capital (USDC)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Drawdown
        cummax = self.equity_curve['equity'].cummax()
        drawdown = (self.equity_curve['equity'] - cummax) / cummax * 100
        axes[0, 1].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        axes[0, 1].set_title('Drawdown', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # PnL distribution
        axes[1, 0].hist(trades_df['pnl'], bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_title('PnL Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('PnL (USDC)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        # Cumulative PnL
        cumulative_pnl = trades_df['pnl'].cumsum()
        axes[1, 1].plot(cumulative_pnl, linewidth=1.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Cumulative PnL', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('Cumulative PnL (USDC)')
        axes[1, 1].grid(True, alpha=0.3)

        # Win/Loss by position type
        long_trades = trades_df[trades_df['position'] == 'long']
        short_trades = trades_df[trades_df['position'] == 'short']

        position_stats = pd.DataFrame({
            'Long': [len(long_trades[long_trades['pnl'] > 0]), len(long_trades[long_trades['pnl'] <= 0])],
            'Short': [len(short_trades[short_trades['pnl'] > 0]), len(short_trades[short_trades['pnl'] <= 0])]
        }, index=['Wins', 'Losses'])

        position_stats.plot(kind='bar', ax=axes[2, 0], color=['green', 'red'], alpha=0.7)
        axes[2, 0].set_title('Wins/Losses by Position Type', fontsize=12, fontweight='bold')
        axes[2, 0].set_ylabel('Count')
        axes[2, 0].set_xticklabels(['Wins', 'Losses'], rotation=0)
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # Signal strength vs PnL
        axes[2, 1].scatter(trades_df['signal_strength'], trades_df['pnl'], alpha=0.5, s=10)
        axes[2, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[2, 1].set_title('Signal Strength vs PnL', fontsize=12, fontweight='bold')
        axes[2, 1].set_xlabel('Signal Strength')
        axes[2, 1].set_ylabel('PnL (USDC)')
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Backtest plots saved to {save_path}")

    def export_trades(self, path='outputs/trades.csv'):
        """Export trades to CSV"""
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(path, index=False)
            print(f"Trades exported to {path}")


if __name__ == "__main__":
    # Test backtesting
    from model_training import ModelTrainer
    from feature_engineering import get_feature_columns

    # Load data
    df = pd.read_csv('data/data_with_features.csv', index_col=0, parse_dates=True)

    # Load model
    trainer = ModelTrainer()
    trainer.load_model('models/lightgbm_model.pkl')

    # Get feature columns
    feature_cols = get_feature_columns(df)

    # Split data
    X_train, X_test, X_oot, y_train, y_test, y_oot, train_idx, test_idx, oot_idx = trainer.split_data(df, feature_cols)

    # Get predictions for OOT
    predictions = trainer.model.predict(X_oot)

    # Get labels and signal strengths for OOT
    oot_data = df.loc[oot_idx]
    labels = oot_data['label']
    signal_strengths = oot_data['signal_strength']

    # Run backtest
    os.makedirs('outputs', exist_ok=True)
    backtester = Backtester()
    metrics = backtester.run_backtest(oot_data, predictions, labels, signal_strengths)

    # Plot results
    backtester.plot_results()

    # Export trades
    backtester.export_trades()
