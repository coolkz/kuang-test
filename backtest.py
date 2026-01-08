"""
Backtesting Framework
Evaluates trading signals and computes performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime


class Backtester:
    def __init__(self, data, signals, initial_capital=100000):
        """
        Initialize the backtester

        Args:
            data (pd.DataFrame): Historical price data
            signals (pd.DataFrame): Trading signals
            initial_capital (float): Initial capital for backtesting
        """
        self.data = data.copy()
        self.signals = signals.copy()
        self.initial_capital = initial_capital
        self.results = {}

    def backtest_signal(self, signal_name, holding_period=5, transaction_cost=0.001):
        """
        Backtest a specific signal

        Args:
            signal_name (str): Name of the signal column
            holding_period (int): Number of days to hold position
            transaction_cost (float): Transaction cost as percentage (0.001 = 0.1%)

        Returns:
            dict: Backtesting results
        """
        if signal_name not in self.signals.columns:
            raise ValueError(f"Signal '{signal_name}' not found")

        print(f"\nBacktesting signal: {signal_name}")

        # Get signal occurrences
        signal = self.signals[signal_name]
        signal_dates = signal[signal == 1].index

        if len(signal_dates) == 0:
            print(f"No signals found for {signal_name}")
            return None

        # Track trades
        trades = []
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        position = 0
        position_entry_price = 0
        position_entry_date = None

        for date in self.data.index:
            close_price = self.data.loc[date, 'Close']

            # Check if we have an open position that should be closed
            if position > 0 and position_entry_date is not None:
                days_held = (date - position_entry_date).days
                if days_held >= holding_period:
                    # Close position
                    exit_price = close_price
                    exit_value = position * exit_price * (1 - transaction_cost)
                    pnl = exit_value - (position * position_entry_price)
                    pnl_pct = (exit_price / position_entry_price - 1) * 100

                    trades.append({
                        'entry_date': position_entry_date,
                        'entry_price': position_entry_price,
                        'exit_date': date,
                        'exit_price': exit_price,
                        'shares': position,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'holding_days': days_held
                    })

                    cash = exit_value
                    position = 0
                    position_entry_price = 0
                    position_entry_date = None

            # Check for new signal (only if no open position)
            if date in signal_dates and position == 0:
                # Open position
                entry_price = close_price
                position = int(cash / (entry_price * (1 + transaction_cost)))
                cost = position * entry_price * (1 + transaction_cost)

                if position > 0:
                    cash -= cost
                    position_entry_price = entry_price
                    position_entry_date = date

        # Close any remaining position at the end
        if position > 0:
            exit_price = self.data.iloc[-1]['Close']
            exit_value = position * exit_price * (1 - transaction_cost)
            pnl = exit_value - (position * position_entry_price)
            pnl_pct = (exit_price / position_entry_price - 1) * 100

            trades.append({
                'entry_date': position_entry_date,
                'entry_price': position_entry_price,
                'exit_date': self.data.index[-1],
                'exit_price': exit_price,
                'shares': position,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'holding_days': (self.data.index[-1] - position_entry_date).days
            })

            cash = exit_value

        # Calculate metrics
        if len(trades) == 0:
            print(f"No completed trades for {signal_name}")
            return None

        trades_df = pd.DataFrame(trades)

        # Performance metrics
        total_return = (cash - self.initial_capital) / self.initial_capital * 100
        num_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / num_trades * 100 if num_trades > 0 else 0

        avg_return = trades_df['pnl_pct'].mean()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0

        max_win = trades_df['pnl_pct'].max()
        max_loss = trades_df['pnl_pct'].min()

        # Risk metrics
        profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() /
                           trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else float('inf')

        # Sharpe ratio (simplified)
        if trades_df['pnl_pct'].std() > 0:
            sharpe = trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std() * np.sqrt(252 / holding_period)
        else:
            sharpe = 0

        # Maximum drawdown
        cumulative_returns = (1 + trades_df['pnl_pct'] / 100).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        results = {
            'signal_name': signal_name,
            'total_return_pct': total_return,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'avg_return_pct': avg_return,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'max_win_pct': max_win,
            'max_loss_pct': max_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'final_capital': cash,
            'trades': trades_df
        }

        self.results[signal_name] = results
        return results

    def backtest_all_signals(self, holding_period=5, transaction_cost=0.001, top_n=None):
        """
        Backtest all signals

        Args:
            holding_period (int): Number of days to hold position
            transaction_cost (float): Transaction cost as percentage
            top_n (int): Only backtest top N signals by frequency (None = all)

        Returns:
            pd.DataFrame: Summary of all backtests
        """
        print("=" * 80)
        print("BACKTESTING ALL SIGNALS")
        print("=" * 80)

        # Optionally filter to most frequent signals
        if top_n:
            signal_counts = self.signals.sum().sort_values(ascending=False)
            signals_to_test = signal_counts.head(top_n).index.tolist()
        else:
            signals_to_test = self.signals.columns.tolist()

        summary_data = []

        for signal_name in signals_to_test:
            try:
                result = self.backtest_signal(signal_name, holding_period, transaction_cost)
                if result:
                    summary_data.append({
                        'signal': result['signal_name'],
                        'total_return': result['total_return_pct'],
                        'num_trades': result['num_trades'],
                        'win_rate': result['win_rate_pct'],
                        'avg_return': result['avg_return_pct'],
                        'sharpe': result['sharpe_ratio'],
                        'max_drawdown': result['max_drawdown_pct'],
                        'profit_factor': result['profit_factor']
                    })
            except Exception as e:
                print(f"Error backtesting {signal_name}: {e}")
                continue

        if len(summary_data) == 0:
            print("No successful backtests")
            return pd.DataFrame()

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('total_return', ascending=False)

        return summary_df

    def print_backtest_results(self, signal_name):
        """
        Print detailed results for a specific signal

        Args:
            signal_name (str): Name of the signal
        """
        if signal_name not in self.results:
            print(f"No results found for {signal_name}")
            return

        result = self.results[signal_name]

        print("\n" + "=" * 80)
        print(f"BACKTEST RESULTS: {signal_name}")
        print("=" * 80)

        print(f"\nPerformance Metrics:")
        print(f"  Total Return:      {result['total_return_pct']:>10.2f}%")
        print(f"  Final Capital:     ${result['final_capital']:>10,.2f}")
        print(f"  Initial Capital:   ${self.initial_capital:>10,.2f}")

        print(f"\nTrade Statistics:")
        print(f"  Number of Trades:  {result['num_trades']:>10}")
        print(f"  Winning Trades:    {result['winning_trades']:>10}")
        print(f"  Losing Trades:     {result['losing_trades']:>10}")
        print(f"  Win Rate:          {result['win_rate_pct']:>10.2f}%")

        print(f"\nReturn Metrics:")
        print(f"  Average Return:    {result['avg_return_pct']:>10.2f}%")
        print(f"  Average Win:       {result['avg_win_pct']:>10.2f}%")
        print(f"  Average Loss:      {result['avg_loss_pct']:>10.2f}%")
        print(f"  Max Win:           {result['max_win_pct']:>10.2f}%")
        print(f"  Max Loss:          {result['max_loss_pct']:>10.2f}%")

        print(f"\nRisk Metrics:")
        print(f"  Profit Factor:     {result['profit_factor']:>10.2f}")
        print(f"  Sharpe Ratio:      {result['sharpe_ratio']:>10.2f}")
        print(f"  Max Drawdown:      {result['max_drawdown_pct']:>10.2f}%")

        print(f"\nRecent Trades:")
        print(result['trades'].tail(10).to_string(index=False))

    def get_best_signals(self, metric='total_return_pct', top_n=10):
        """
        Get the best performing signals

        Args:
            metric (str): Metric to rank by
            top_n (int): Number of top signals to return

        Returns:
            list: List of top signal names
        """
        if len(self.results) == 0:
            return []

        rankings = [(name, result[metric]) for name, result in self.results.items()]
        rankings.sort(key=lambda x: x[1], reverse=True)

        return [name for name, _ in rankings[:top_n]]


if __name__ == "__main__":
    # Test the backtester
    from data_fetcher import DataFetcher
    from factors import FactorEngine
    from signals import SignalGenerator

    print("Testing Backtester...")

    fetcher = DataFetcher("META")
    data = fetcher.fetch_data(period="2y")

    engine = FactorEngine(data)
    factors = engine.compute_all_factors()

    generator = SignalGenerator(data, factors)
    signals = generator.generate_all_signals()

    backtester = Backtester(data, signals, initial_capital=100000)

    # Backtest a few key signals
    test_signals = ['golden_cross', 'rsi_oversold_30', 'trend_following', 'mean_reversion_buy']

    for signal in test_signals:
        if signal in signals.columns:
            backtester.backtest_signal(signal, holding_period=5)
            backtester.print_backtest_results(signal)
