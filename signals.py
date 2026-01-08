"""
Signal Generation Module
Generates trading signals based on factors and strategies
"""

import pandas as pd
import numpy as np


class SignalGenerator:
    def __init__(self, data, factors):
        """
        Initialize the signal generator

        Args:
            data (pd.DataFrame): Historical price data
            factors (pd.DataFrame): Computed factors
        """
        self.data = data.copy()
        self.factors = factors.copy()
        self.signals = pd.DataFrame(index=self.data.index)

    def generate_all_signals(self):
        """
        Generate all available trading signals

        Returns:
            pd.DataFrame: DataFrame with all signals
        """
        print("Generating trading signals...")

        # Trend-following signals
        self.generate_moving_average_signals()
        self.generate_macd_signals()

        # Mean-reversion signals
        self.generate_rsi_signals()
        self.generate_bollinger_signals()

        # Momentum signals
        self.generate_momentum_signals()

        # Volume signals
        self.generate_volume_signals()

        # Composite signals
        self.generate_composite_signals()

        print(f"Generated {len(self.signals.columns)} signals")
        return self.signals

    def generate_moving_average_signals(self):
        """Generate moving average crossover signals"""

        # Golden Cross / Death Cross (50/200)
        sma_50 = self.factors['sma_50']
        sma_200 = self.factors['sma_200']
        self.signals['golden_cross'] = ((sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))).astype(int)
        self.signals['death_cross'] = ((sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1))).astype(int)

        # EMA crossover (12/26)
        ema_12 = self.factors['ema_12']
        ema_26 = self.factors['ema_26']
        self.signals['ema_cross_long'] = ((ema_12 > ema_26) & (ema_12.shift(1) <= ema_26.shift(1))).astype(int)
        self.signals['ema_cross_short'] = ((ema_12 < ema_26) & (ema_12.shift(1) >= ema_26.shift(1))).astype(int)

        # Price above/below moving averages
        close = self.data['Close']
        self.signals['price_above_sma_50'] = (close > sma_50).astype(int)
        self.signals['price_above_sma_200'] = (close > sma_200).astype(int)

        # Distance-based signals
        dist_sma_20 = self.factors['dist_sma_20']
        self.signals['sma_20_oversold'] = (dist_sma_20 < -0.05).astype(int)  # 5% below
        self.signals['sma_20_overbought'] = (dist_sma_20 > 0.05).astype(int)  # 5% above

    def generate_macd_signals(self):
        """Generate MACD-based signals"""
        macd = self.factors['macd']
        macd_signal = self.factors['macd_signal']
        macd_diff = self.factors['macd_diff']

        # MACD crossover
        self.signals['macd_bullish'] = ((macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))).astype(int)
        self.signals['macd_bearish'] = ((macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))).astype(int)

        # MACD histogram divergence
        self.signals['macd_increasing'] = (macd_diff > macd_diff.shift(1)).astype(int)
        self.signals['macd_decreasing'] = (macd_diff < macd_diff.shift(1)).astype(int)

    def generate_rsi_signals(self):
        """Generate RSI-based signals"""
        rsi_14 = self.factors['rsi_14']

        # Traditional RSI levels
        self.signals['rsi_oversold_30'] = (rsi_14 < 30).astype(int)
        self.signals['rsi_overbought_70'] = (rsi_14 > 70).astype(int)

        # More conservative levels
        self.signals['rsi_oversold_40'] = (rsi_14 < 40).astype(int)
        self.signals['rsi_overbought_60'] = (rsi_14 > 60).astype(int)

        # RSI momentum
        self.signals['rsi_rising'] = ((rsi_14 > rsi_14.shift(1)) & (rsi_14.shift(1) > rsi_14.shift(2))).astype(int)
        self.signals['rsi_falling'] = ((rsi_14 < rsi_14.shift(1)) & (rsi_14.shift(1) < rsi_14.shift(2))).astype(int)

        # RSI divergence (simplified)
        price_change = self.data['Close'].pct_change(5)
        rsi_change = rsi_14.diff(5)
        self.signals['rsi_bullish_divergence'] = ((price_change < 0) & (rsi_change > 0) & (rsi_14 < 40)).astype(int)
        self.signals['rsi_bearish_divergence'] = ((price_change > 0) & (rsi_change < 0) & (rsi_14 > 60)).astype(int)

    def generate_bollinger_signals(self):
        """Generate Bollinger Bands signals"""
        bb_position = self.factors['bb_position']
        bb_width = self.factors['bb_width']

        # Touch or break bands
        self.signals['bb_lower_touch'] = (bb_position < 0.1).astype(int)  # Near lower band
        self.signals['bb_upper_touch'] = (bb_position > 0.9).astype(int)  # Near upper band

        # Band squeeze (low volatility)
        bb_width_sma = bb_width.rolling(window=20).mean()
        self.signals['bb_squeeze'] = (bb_width < bb_width_sma * 0.7).astype(int)

        # Band expansion (high volatility)
        self.signals['bb_expansion'] = (bb_width > bb_width_sma * 1.3).astype(int)

        # Mean reversion
        close = self.data['Close']
        bb_mid = self.factors['bb_mid']
        self.signals['bb_mean_revert_long'] = ((close < bb_mid) & (close.shift(1) >= bb_mid.shift(1))).astype(int)
        self.signals['bb_mean_revert_short'] = ((close > bb_mid) & (close.shift(1) <= bb_mid.shift(1))).astype(int)

    def generate_momentum_signals(self):
        """Generate momentum-based signals"""

        # Rate of Change signals
        roc_10 = self.factors['roc_10']
        self.signals['roc_strong_positive'] = (roc_10 > 5).astype(int)  # 5% gain in 10 days
        self.signals['roc_strong_negative'] = (roc_10 < -5).astype(int)  # 5% loss in 10 days

        # Multi-period momentum
        return_5d = self.factors['return_5d']
        return_10d = self.factors['return_10d']
        return_20d = self.factors['return_20d']

        self.signals['momentum_aligned_bullish'] = ((return_5d > 0) & (return_10d > 0) & (return_20d > 0)).astype(int)
        self.signals['momentum_aligned_bearish'] = ((return_5d < 0) & (return_10d < 0) & (return_20d < 0)).astype(int)

        # Stochastic signals
        stoch_k = self.factors['stoch_k']
        stoch_d = self.factors['stoch_d']

        self.signals['stoch_oversold'] = ((stoch_k < 20) & (stoch_d < 20)).astype(int)
        self.signals['stoch_overbought'] = ((stoch_k > 80) & (stoch_d > 80)).astype(int)
        self.signals['stoch_bullish_cross'] = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))).astype(int)

    def generate_volume_signals(self):
        """Generate volume-based signals"""
        volume_ratio_5d = self.factors['volume_ratio_5d']
        volume_ratio_20d = self.factors['volume_ratio_20d']

        # High volume breakouts
        self.signals['volume_spike'] = (volume_ratio_5d > 2.0).astype(int)  # 2x average volume
        self.signals['volume_surge'] = (volume_ratio_20d > 1.5).astype(int)

        # OBV trend
        obv = self.factors['obv']
        obv_ema = self.factors['obv_ema']
        self.signals['obv_bullish'] = (obv > obv_ema).astype(int)
        self.signals['obv_bearish'] = (obv < obv_ema).astype(int)

        # Volume + Price confirmation
        close = self.data['Close']
        price_up = (close > close.shift(1)).astype(int)
        price_down = (close < close.shift(1)).astype(int)

        self.signals['volume_price_confirm_bull'] = (price_up & (volume_ratio_5d > 1.2)).astype(int)
        self.signals['volume_price_confirm_bear'] = (price_down & (volume_ratio_5d > 1.2)).astype(int)

    def generate_composite_signals(self):
        """Generate composite signals combining multiple factors"""

        # Strong Bullish Signal
        self.signals['strong_bullish'] = (
            (self.signals['price_above_sma_50'] == 1) &
            (self.signals['price_above_sma_200'] == 1) &
            (self.signals['rsi_oversold_40'] == 1) &
            (self.signals['macd_increasing'] == 1)
        ).astype(int)

        # Strong Bearish Signal
        self.signals['strong_bearish'] = (
            (self.signals['price_above_sma_50'] == 0) &
            (self.signals['price_above_sma_200'] == 0) &
            (self.signals['rsi_overbought_60'] == 1) &
            (self.signals['macd_decreasing'] == 1)
        ).astype(int)

        # Mean Reversion Buy Signal
        self.signals['mean_reversion_buy'] = (
            (self.signals['rsi_oversold_30'] == 1) &
            (self.signals['bb_lower_touch'] == 1) &
            (self.signals['volume_spike'] == 1)
        ).astype(int)

        # Momentum Breakout Signal
        self.signals['momentum_breakout'] = (
            (self.signals['golden_cross'] == 1) |
            (self.signals['ema_cross_long'] == 1)
        ) & (
            self.signals['volume_spike'] == 1
        ).astype(int)

        # Trend Following Signal
        self.signals['trend_following'] = (
            (self.signals['price_above_sma_50'] == 1) &
            (self.signals['price_above_sma_200'] == 1) &
            (self.signals['macd_bullish'] == 1)
        ).astype(int)

    def get_signals(self):
        """
        Get the generated signals

        Returns:
            pd.DataFrame: DataFrame with all signals
        """
        return self.signals

    def get_active_signals(self, date=None):
        """
        Get active signals for a specific date

        Args:
            date: Date to check (defaults to last date)

        Returns:
            dict: Active signals
        """
        if date is None:
            date = self.signals.index[-1]

        active = {}
        for col in self.signals.columns:
            if self.signals.loc[date, col] == 1:
                active[col] = 1

        return active

    def get_signal_summary(self):
        """
        Get summary statistics of signals

        Returns:
            pd.DataFrame: Summary of signal occurrences
        """
        summary = pd.DataFrame({
            'signal': self.signals.columns,
            'count': [self.signals[col].sum() for col in self.signals.columns],
            'frequency': [self.signals[col].mean() for col in self.signals.columns]
        })
        return summary.sort_values('count', ascending=False)


if __name__ == "__main__":
    # Test the signal generator
    from data_fetcher import DataFetcher
    from factors import FactorEngine

    fetcher = DataFetcher("META")
    data = fetcher.fetch_data(period="1y")

    engine = FactorEngine(data)
    factors = engine.compute_all_factors()

    generator = SignalGenerator(data, factors)
    signals = generator.generate_all_signals()

    print("\nSignal preview:")
    print(signals.tail())

    print("\nSignal summary:")
    print(generator.get_signal_summary().head(20))

    print("\nActive signals (last date):")
    active = generator.get_active_signals()
    for signal, value in active.items():
        print(f"  {signal}")
