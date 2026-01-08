"""
Factors and Technical Indicators Module
Computes various quantitative factors and indicators for trading signals
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


class FactorEngine:
    def __init__(self, data):
        """
        Initialize the factor engine

        Args:
            data (pd.DataFrame): Historical price data with OHLCV columns
        """
        self.data = data.copy()
        self.factors = pd.DataFrame(index=self.data.index)

    def compute_all_factors(self):
        """
        Compute all available factors

        Returns:
            pd.DataFrame: DataFrame with all computed factors
        """
        print("Computing factors...")

        # Price-based factors
        self.compute_returns()
        self.compute_moving_averages()
        self.compute_momentum_factors()

        # Volatility factors
        self.compute_volatility_factors()

        # Volume factors
        self.compute_volume_factors()

        # Pattern recognition factors
        self.compute_pattern_factors()

        # Statistical factors
        self.compute_statistical_factors()

        print(f"Computed {len(self.factors.columns)} factors")
        return self.factors

    def compute_returns(self):
        """Compute various return metrics"""
        close = self.data['Close']

        # Simple returns
        self.factors['return_1d'] = close.pct_change(1)
        self.factors['return_5d'] = close.pct_change(5)
        self.factors['return_10d'] = close.pct_change(10)
        self.factors['return_20d'] = close.pct_change(20)

        # Log returns
        self.factors['log_return_1d'] = np.log(close / close.shift(1))

    def compute_moving_averages(self):
        """Compute moving average based factors"""
        close = self.data['Close']

        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            sma = SMAIndicator(close=close, window=period)
            self.factors[f'sma_{period}'] = sma.sma_indicator()
            # Distance from moving average (normalized)
            self.factors[f'dist_sma_{period}'] = (close - self.factors[f'sma_{period}']) / self.factors[f'sma_{period}']

        # Exponential Moving Averages
        for period in [12, 26, 50]:
            ema = EMAIndicator(close=close, window=period)
            self.factors[f'ema_{period}'] = ema.ema_indicator()

        # Moving Average Crossovers
        self.factors['sma_cross_50_200'] = (self.factors['sma_50'] - self.factors['sma_200']) / self.factors['sma_200']
        self.factors['ema_cross_12_26'] = (self.factors['ema_12'] - self.factors['ema_26']) / self.factors['ema_26']

        # MACD
        macd = MACD(close=close)
        self.factors['macd'] = macd.macd()
        self.factors['macd_signal'] = macd.macd_signal()
        self.factors['macd_diff'] = macd.macd_diff()

    def compute_momentum_factors(self):
        """Compute momentum-based factors"""
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']

        # RSI
        for period in [14, 21]:
            rsi = RSIIndicator(close=close, window=period)
            self.factors[f'rsi_{period}'] = rsi.rsi()

        # Stochastic Oscillator
        stoch = StochasticOscillator(high=high, low=low, close=close)
        self.factors['stoch_k'] = stoch.stoch()
        self.factors['stoch_d'] = stoch.stoch_signal()

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            self.factors[f'roc_{period}'] = ((close - close.shift(period)) / close.shift(period)) * 100

    def compute_volatility_factors(self):
        """Compute volatility-based factors"""
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']

        # Historical Volatility
        returns = close.pct_change()
        for period in [10, 20, 30]:
            self.factors[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)

        # Bollinger Bands
        bb = BollingerBands(close=close, window=20, window_dev=2)
        self.factors['bb_high'] = bb.bollinger_hband()
        self.factors['bb_low'] = bb.bollinger_lband()
        self.factors['bb_mid'] = bb.bollinger_mavg()
        self.factors['bb_width'] = bb.bollinger_wband()
        self.factors['bb_position'] = (close - self.factors['bb_low']) / (self.factors['bb_high'] - self.factors['bb_low'])

        # Average True Range
        atr = AverageTrueRange(high=high, low=low, close=close, window=14)
        self.factors['atr'] = atr.average_true_range()
        self.factors['atr_pct'] = self.factors['atr'] / close

    def compute_volume_factors(self):
        """Compute volume-based factors"""
        close = self.data['Close']
        volume = self.data['Volume']
        high = self.data['High']
        low = self.data['Low']

        # Volume ratios
        self.factors['volume_ratio_5d'] = volume / volume.rolling(window=5).mean()
        self.factors['volume_ratio_20d'] = volume / volume.rolling(window=20).mean()

        # On-Balance Volume
        obv = OnBalanceVolumeIndicator(close=close, volume=volume)
        self.factors['obv'] = obv.on_balance_volume()
        self.factors['obv_ema'] = self.factors['obv'].ewm(span=20).mean()

        # VWAP
        vwap = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume)
        self.factors['vwap'] = vwap.volume_weighted_average_price()
        self.factors['dist_vwap'] = (close - self.factors['vwap']) / self.factors['vwap']

    def compute_pattern_factors(self):
        """Compute price pattern factors"""
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        open_price = self.data['Open']

        # Price range
        self.factors['high_low_range'] = (high - low) / close
        self.factors['close_open_range'] = (close - open_price) / open_price

        # Position in day's range
        self.factors['position_in_range'] = (close - low) / (high - low)

        # Higher highs and lower lows
        self.factors['higher_high_5d'] = (high > high.shift(1).rolling(window=5).max()).astype(int)
        self.factors['lower_low_5d'] = (low < low.shift(1).rolling(window=5).min()).astype(int)

        # Distance from recent high/low
        self.factors['dist_from_high_20d'] = (high.rolling(window=20).max() - close) / close
        self.factors['dist_from_low_20d'] = (close - low.rolling(window=20).min()) / close

    def compute_statistical_factors(self):
        """Compute statistical factors"""
        close = self.data['Close']
        returns = close.pct_change()

        # Z-scores
        for period in [20, 50]:
            mean = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            self.factors[f'zscore_{period}'] = (close - mean) / std

        # Skewness and Kurtosis of returns
        for period in [20, 50]:
            self.factors[f'skew_{period}'] = returns.rolling(window=period).skew()
            self.factors[f'kurt_{period}'] = returns.rolling(window=period).kurt()

        # Sharpe ratio (rolling)
        for period in [20, 50]:
            rolling_return = returns.rolling(window=period).mean()
            rolling_std = returns.rolling(window=period).std()
            self.factors[f'sharpe_{period}'] = (rolling_return / rolling_std) * np.sqrt(252)

    def get_factors(self):
        """
        Get the computed factors

        Returns:
            pd.DataFrame: DataFrame with all factors
        """
        return self.factors

    def get_factor_summary(self):
        """
        Get summary of computed factors

        Returns:
            pd.DataFrame: Summary statistics of factors
        """
        return self.factors.describe()


if __name__ == "__main__":
    # Test the factor engine
    from data_fetcher import DataFetcher

    fetcher = DataFetcher("META")
    data = fetcher.fetch_data(period="1y")

    engine = FactorEngine(data)
    factors = engine.compute_all_factors()

    print("\nFactor preview:")
    print(factors.head())
    print(f"\nTotal factors: {len(factors.columns)}")
    print("\nFactor names:")
    print(factors.columns.tolist())
