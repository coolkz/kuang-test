"""
Data Fetcher Module for Meta Stock
Fetches historical price data and prepares it for analysis
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class DataFetcher:
    def __init__(self, ticker="META"):
        """
        Initialize the data fetcher

        Args:
            ticker (str): Stock ticker symbol (default: META)
        """
        self.ticker = ticker
        self.data = None

    def fetch_data(self, start_date=None, end_date=None, period="2y"):
        """
        Fetch historical stock data

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            period (str): Period to fetch if dates not specified (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            pd.DataFrame: Historical stock data
        """
        print(f"Fetching data for {self.ticker}...")

        try:
            if start_date and end_date:
                self.data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            else:
                self.data = yf.download(self.ticker, period=period, progress=False)

            if self.data.empty:
                raise ValueError(f"No data found for {self.ticker}")

            # Flatten multi-level columns if present
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = self.data.columns.get_level_values(0)

            print(f"Successfully fetched {len(self.data)} rows of data")
            print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")

            return self.data

        except Exception as e:
            print(f"Error fetching data: {e}")
            raise

    def get_data(self):
        """
        Get the fetched data

        Returns:
            pd.DataFrame: Historical stock data
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        return self.data

    def get_summary_stats(self):
        """
        Get summary statistics of the data

        Returns:
            dict: Summary statistics
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")

        stats = {
            'ticker': self.ticker,
            'start_date': str(self.data.index[0]),
            'end_date': str(self.data.index[-1]),
            'num_days': len(self.data),
            'avg_price': float(self.data['Close'].mean()),
            'min_price': float(self.data['Close'].min()),
            'max_price': float(self.data['Close'].max()),
            'avg_volume': float(self.data['Volume'].mean()),
            'total_return': float((self.data['Close'].iloc[-1] / self.data['Close'].iloc[0] - 1) * 100)
        }

        return stats


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = DataFetcher("META")
    data = fetcher.fetch_data(period="1y")
    print("\nData preview:")
    print(data.head())
    print("\nSummary statistics:")
    stats = fetcher.get_summary_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
