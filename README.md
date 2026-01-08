# Meta Stock Quantitative Trading System

A comprehensive quantitative trading system for finding and evaluating trading signals/factors for Meta (META) stock.

## Overview

This system provides a complete framework for:
- Fetching historical stock data
- Computing 50+ technical indicators and factors
- Generating 40+ trading signals across multiple strategies
- Backtesting signals with detailed performance metrics
- Identifying the best performing trading strategies

## Features

### Data Fetching
- Automatic download of historical price data using yfinance
- Configurable time periods (1 month to max history)
- Summary statistics and data quality checks

### Factor Engineering (50+ Factors)
- **Price-based**: Returns, moving averages, distance metrics
- **Momentum**: RSI, Stochastic, Rate of Change, MACD
- **Volatility**: Bollinger Bands, ATR, historical volatility
- **Volume**: OBV, VWAP, volume ratios
- **Pattern Recognition**: High/low patterns, range metrics
- **Statistical**: Z-scores, skewness, kurtosis, rolling Sharpe

### Signal Generation (40+ Signals)
- **Trend Following**: Golden Cross, Death Cross, EMA crossovers
- **Mean Reversion**: RSI oversold/overbought, Bollinger touches
- **Momentum**: ROC signals, aligned momentum, Stochastic patterns
- **Volume-based**: Volume spikes, volume-price confirmation
- **Composite**: Multi-factor signals combining multiple indicators

### Backtesting Framework
- Realistic transaction costs
- Configurable holding periods
- Comprehensive performance metrics:
  - Total return, win rate, profit factor
  - Sharpe ratio, maximum drawdown
  - Average win/loss statistics
  - Trade-by-trade analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd kuang-test
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the full analysis with default settings (2 years of data, 5-day holding period):

```bash
python main.py
```

### Custom Analysis

Specify different parameters:

```bash
# Analyze 1 year of data with 10-day holding period
python main.py --period 1y --holding-period 10

# Backtest top 30 signals
python main.py --top-signals 30

# Export results to CSV
python main.py --export results.csv
```

### Command Line Options

- `--ticker TICKER`: Stock ticker symbol (default: META)
- `--period PERIOD`: Historical period (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)
- `--holding-period DAYS`: Days to hold each position (default: 5)
- `--top-signals N`: Number of top signals to backtest (default: 20)
- `--export FILE`: Export full results to CSV file

## Module Usage

### Using Individual Modules

```python
from data_fetcher import DataFetcher
from factors import FactorEngine
from signals import SignalGenerator
from backtest import Backtester

# Fetch data
fetcher = DataFetcher("META")
data = fetcher.fetch_data(period="1y")

# Compute factors
engine = FactorEngine(data)
factors = engine.compute_all_factors()

# Generate signals
generator = SignalGenerator(data, factors)
signals = generator.generate_all_signals()

# Backtest a specific signal
backtester = Backtester(data, signals, initial_capital=100000)
results = backtester.backtest_signal('golden_cross', holding_period=5)
backtester.print_backtest_results('golden_cross')
```

## Output Examples

### Summary Output
The system provides:
1. Data summary with price statistics
2. List of computed factors
3. Signal frequency analysis
4. Backtesting performance table
5. Detailed results for top performers
6. Current market analysis with active signals
7. Trading recommendations

### Current Market Analysis
- Latest price and key factor values
- Active signals at the current moment
- Alerts when high-performing signals trigger

### Backtest Results
For each signal, you'll see:
- Total return and final capital
- Number of trades and win rate
- Average returns (overall, wins, losses)
- Risk metrics (Sharpe ratio, max drawdown, profit factor)
- Recent trade history

## Project Structure

```
kuang-test/
├── main.py              # Main program orchestrating the system
├── data_fetcher.py      # Data download and management
├── factors.py           # Technical indicators and factor computation
├── signals.py           # Trading signal generation
├── backtest.py          # Backtesting framework
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Key Signals Explained

### Trend Following Signals
- **golden_cross**: SMA 50 crosses above SMA 200 (bullish)
- **death_cross**: SMA 50 crosses below SMA 200 (bearish)
- **trend_following**: Price above both SMAs with bullish MACD

### Mean Reversion Signals
- **rsi_oversold_30**: RSI below 30 (potential bounce)
- **rsi_overbought_70**: RSI above 70 (potential pullback)
- **mean_reversion_buy**: RSI oversold + BB lower touch + volume spike

### Momentum Signals
- **momentum_breakout**: MA crossover with volume confirmation
- **momentum_aligned_bullish**: Positive returns across 5, 10, 20 days
- **roc_strong_positive**: 5%+ gain in 10 days

### Composite Signals
- **strong_bullish**: Multiple bullish factors aligned
- **strong_bearish**: Multiple bearish factors aligned

## Performance Considerations

- The system computes 50+ factors and 40+ signals
- Backtesting is performed trade-by-trade for accuracy
- For faster results, use `--top-signals` to limit backtests
- Export to CSV for further analysis in Excel or other tools

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Past performance does not guarantee future results. Always do your own research and consult with financial professionals before making investment decisions.

## License

This project is provided as-is for educational purposes.