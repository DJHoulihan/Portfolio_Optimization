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
        self.config = config

    def load_data(self):
        """
        Returns:
            obs_windows : (T, N, K, F)
            returns     : (T, N)
        """
        # Load raw data
        df = pd.read_parquet(self.config.data.data_path) # can change path in config.ini

        df = df.sort_values(["date", "permno"]) # assumes crsp format

        # Fix asset universe 
        min_obs = self.config.data.min_obs
        max_assets = self.config.data.max_assets

        counts = df.groupby("permno")["ret"].count()
        valid_permnos = counts[counts >= min_obs].index

        df = df[df["permno"].isin(valid_permnos)]

        # Optional: cap universe size by market cap
        if max_assets is not None:
            top_permnos = (
                df.groupby("permno")["mktcap"].mean()
                  .sort_values(ascending=False)
                  .head(max_assets)
                  .index
            )
            df = df[df["permno"].isin(top_permnos)]

        permnos = np.sort(df["permno"].unique())
        N = len(permnos)

        # Select features
        exclude_cols = {"date", "permno"}
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        assert "ret" in feature_cols, "`ret` should usually be included as a feature"


        # Pivot to (T_raw, N, F)
        feature_arrays = []

        for col in feature_cols:
            pivot = df.pivot(index="date", columns="permno", values=col)
            pivot = pivot.reindex(columns=permnos)
            feature_arrays.append(pivot.values)

        features = np.stack(feature_arrays, axis=-1)
        # (T_raw, N, F)

        # Returns (separate target)
        ret_df = df.pivot(index="date", columns="permno", values="ret")
        ret_df = ret_df.reindex(columns=permnos)
        returns = ret_df.values
        # (T_raw, N)

        # Handle missing data
        features = np.nan_to_num(features, nan=0.0)
        returns = np.nan_to_num(returns, nan=0.0)

        # Build rolling windows
        K = self.config.training.lookback
        T_raw = features.shape[0]

        windows = np.stack(
            [features[t - K:t] for t in range(K, T_raw)],
            axis=0
        )
        # (T, K, N, F)
        # normalize within windows to avoid scaling issues
        windows = temporal_zscore_normalize(windows)

        obs_windows = windows.transpose(0, 2, 1, 3)
        # (T, N, K, F)

        returns = returns[K:]
        # (T, N)

        # sanity checks
        assert obs_windows.shape == (returns.shape[0], N, K, len(features))
        assert np.all(np.isfinite(obs_windows)), 'obs_window contains nan values'

        assert obs_windows.shape[0] == returns.shape[0]
        assert obs_windows.shape[1] == returns.shape[1]

        print(f"[DataHandler] obs_windows: {obs_windows.shape}")
        print(f"[DataHandler] returns:     {returns.shape}")

        return obs_windows.astype(np.float32), returns.astype(np.float32)



class AlphaVantageData:
    def __init__(self, config):
        self.api_key = config.api.api_key
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


