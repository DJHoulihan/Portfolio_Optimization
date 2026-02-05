import requests
import pandas as pd
import numpy as np
import os
import src.config.config as cf

def temporal_zscore_normalize(X):
    """
    X shape: (samples, time_steps, num_features)
    Returns: normalized_X with same shape
    """
    # mean over time dimension (axis=1)
    mean = np.mean(X, axis=1, keepdims=True)           # shape (samples, 1, features)
    std  = np.std(X, axis=1, keepdims=True) + 1e-8     # avoid division by zero

    return (X - mean) / std

class DataHandler:
    def __init__(self, config):
        # self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.config = config
        self.lookback = config.training.lookback
    
    def load_data(self):
        datafile = self.config.data.data_path / '*.txt'
        return pd.read_csv(datafile, delimiter = ',')
    
    def create_sequences(self, df):
        """
        Converts DataFrame into (num_samples, lookback, num_features) sequences.
        """
        data = df.values
        sequences = []
        for i in range(len(data) - self.lookback):
            sequences.append(temporal_zscore_normalize(data[i:i+self.lookback]))
        return np.array(sequences)  # shape: (num_samples, lookback, num_features)


class AlphaVantageData:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_daily_data(self, symbol, outputsize='compact'):
        """
        Retrieves daily OHLCV data for a symbol from Alpha Vantage.
        """
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key
        }
        r = requests.get(self.base_url, params=params)
        data = r.json()

        if "Time Series (Daily)" not in data:
            raise ValueError(f"No data found for {symbol}: {data}")

        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df = df.sort_index()  # earliest to latest
        df = df.astype(float)

        # optional: select only relevant features
        df = df[['1. open', '2. high', '3. low', '4. close', '6. volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        return df

    def create_sequences(self, df, lookback=20):
        """
        Converts DataFrame into (num_samples, lookback, num_features) sequences.
        """
        data = df.values
        sequences = []
        for i in range(len(data) - lookback):
            sequences.append(data[i:i+lookback])
        return np.array(sequences)  # shape: (num_samples, lookback, num_features)


