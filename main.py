#!/usr/bin/env python3
"""
Meta Stock Quantitative Trading System
Main program to find and evaluate trading signals/factors
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import DataFetcher
from factors import FactorEngine
from signals import SignalGenerator
from backtest import Backtester


def print_header():
    """Print program header"""
    print("\n" + "=" * 80)
    print(" " * 20 + "META STOCK QUANTITATIVE TRADING SYSTEM")
    print("=" * 80 + "\n")


def analyze_current_signals(data, factors, signals):
    """
    Analyze current active signals

    Args:
        data: Price data
        factors: Computed factors
        signals: Generated signals
    """
    print("\n" + "=" * 80)
    print("CURRENT MARKET ANALYSIS")
    print("=" * 80)

    latest_date = data.index[-1]
    latest_close = data['Close'].iloc[-1]

    print(f"\nDate: {latest_date.strftime('%Y-%m-%d')}")
    print(f"Close Price: ${latest_close:.2f}")

    # Get active signals
    generator = SignalGenerator(data, factors)
    generator.signals = signals
    active = generator.get_active_signals()

    if len(active) == 0:
        print("\nNo active signals at the moment.")
    else:
        print(f"\n{len(active)} ACTIVE SIGNALS:")
        for signal in sorted(active.keys()):
            print(f"  • {signal}")

    # Key factor values
    print("\nKEY FACTOR VALUES:")
    latest_factors = factors.iloc[-1]

    key_factors = [
        ('RSI (14)', 'rsi_14'),
        ('MACD', 'macd'),
        ('MACD Signal', 'macd_signal'),
        ('SMA 50', 'sma_50'),
        ('SMA 200', 'sma_200'),
        ('Bollinger Position', 'bb_position'),
        ('Volume Ratio (5d)', 'volume_ratio_5d'),
        ('Volatility (20d)', 'volatility_20'),
    ]

    for name, key in key_factors:
        if key in latest_factors.index:
            value = latest_factors[key]
            if pd.notna(value):
                print(f"  {name:<25} {value:>12.2f}")


def run_full_analysis(ticker="META", period="2y", holding_period=5, top_signals=20):
    """
    Run full quantitative analysis

    Args:
        ticker: Stock ticker
        period: Historical period to analyze
        holding_period: Days to hold each position
        top_signals: Number of top signals to backtest
    """
    print_header()

    # Step 1: Fetch Data
    print("STEP 1: FETCHING DATA")
    print("-" * 80)
    fetcher = DataFetcher(ticker)
    data = fetcher.fetch_data(period=period)
    stats = fetcher.get_summary_stats()

    print(f"\nData Summary:")
    print(f"  Period: {stats['start_date']} to {stats['end_date']}")
    print(f"  Trading Days: {stats['num_days']}")
    print(f"  Price Range: ${stats['min_price']:.2f} - ${stats['max_price']:.2f}")
    print(f"  Total Return: {stats['total_return']:.2f}%")

    # Step 2: Compute Factors
    print("\n" + "=" * 80)
    print("STEP 2: COMPUTING FACTORS")
    print("-" * 80)
    engine = FactorEngine(data)
    factors = engine.compute_all_factors()
    print(f"Computed {len(factors.columns)} factors")

    # Step 3: Generate Signals
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING SIGNALS")
    print("-" * 80)
    generator = SignalGenerator(data, factors)
    signals = generator.generate_all_signals()

    signal_summary = generator.get_signal_summary()
    print(f"\nTop 10 Most Frequent Signals:")
    print(signal_summary.head(10).to_string(index=False))

    # Step 4: Backtest Signals
    print("\n" + "=" * 80)
    print("STEP 4: BACKTESTING SIGNALS")
    print("-" * 80)
    backtester = Backtester(data, signals, initial_capital=100000)
    summary = backtester.backtest_all_signals(
        holding_period=holding_period,
        transaction_cost=0.001,
        top_n=top_signals
    )

    if not summary.empty:
        print("\n" + "=" * 80)
        print("BACKTEST SUMMARY - TOP PERFORMING SIGNALS")
        print("=" * 80)
        print(summary.head(15).to_string(index=False))

        # Detailed results for top 3 signals
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS - TOP 3 SIGNALS")
        print("=" * 80)

        for i, signal in enumerate(summary.head(3)['signal'].values):
            backtester.print_backtest_results(signal)

    # Step 5: Current Market Analysis
    analyze_current_signals(data, factors, signals)

    # Step 6: Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if not summary.empty:
        best_signals = summary.head(5)

        print("\nTop 5 Best Performing Signals:")
        for idx, row in best_signals.iterrows():
            print(f"\n{idx + 1}. {row['signal']}")
            print(f"   Total Return: {row['total_return']:.2f}%")
            print(f"   Win Rate: {row['win_rate']:.2f}%")
            print(f"   Sharpe Ratio: {row['sharpe']:.2f}")
            print(f"   Number of Trades: {int(row['num_trades'])}")

        # Check if any of the best signals are currently active
        generator_check = SignalGenerator(data, factors)
        generator_check.signals = signals
        active_now = generator_check.get_active_signals()

        best_signal_names = set(best_signals['signal'].values)
        active_best = best_signal_names.intersection(set(active_now.keys()))

        if active_best:
            print("\n" + "!" * 80)
            print("ALERT: The following HIGH-PERFORMING signals are ACTIVE NOW:")
            for signal in active_best:
                print(f"  → {signal}")
            print("!" * 80)
        else:
            print("\nNote: None of the top performing signals are currently active.")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


def export_results(ticker="META", period="2y", output_file="results.csv"):
    """
    Export analysis results to CSV

    Args:
        ticker: Stock ticker
        period: Historical period
        output_file: Output CSV file name
    """
    print(f"Exporting results to {output_file}...")

    fetcher = DataFetcher(ticker)
    data = fetcher.fetch_data(period=period)

    engine = FactorEngine(data)
    factors = engine.compute_all_factors()

    generator = SignalGenerator(data, factors)
    signals = generator.generate_all_signals()

    # Combine data, factors, and signals
    results = pd.concat([data, factors, signals], axis=1)
    results.to_csv(output_file)

    print(f"Exported {len(results)} rows to {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Meta Stock Quantitative Trading System - Find and evaluate trading signals"
    )
    parser.add_argument(
        '--ticker',
        type=str,
        default='META',
        help='Stock ticker symbol (default: META)'
    )
    parser.add_argument(
        '--period',
        type=str,
        default='2y',
        choices=['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'],
        help='Historical period to analyze (default: 2y)'
    )
    parser.add_argument(
        '--holding-period',
        type=int,
        default=5,
        help='Number of days to hold each position (default: 5)'
    )
    parser.add_argument(
        '--top-signals',
        type=int,
        default=20,
        help='Number of top signals to backtest (default: 20)'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to CSV file'
    )

    args = parser.parse_args()

    try:
        if args.export:
            export_results(args.ticker, args.period, args.export)
        else:
            run_full_analysis(
                ticker=args.ticker,
                period=args.period,
                holding_period=args.holding_period,
                top_signals=args.top_signals
            )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
