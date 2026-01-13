import requests
import pandas as pd
import numpy as np

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
