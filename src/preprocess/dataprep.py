import numpy as np
<<<<<<< Updated upstream
import pandas as pd
from sklearn.preprocessing import StandardScaler
#
def preprocess(df: pd.DataFrame, drop_cols=None):
    df = df.copy()
=======
# import pandas as pd

def build_features(raw_features):
    """
    raw_features: (N, T, F)
    Assumed to be causal features
    """
    features = raw_features.astype(np.float32)

    # Replace NaNs using forward fill across time
    for i in range(features.shape[0]):
        for f in range(features.shape[2]):
            x = features[i, :, f]
            mask = np.isnan(x)
            if mask.any():
                x[mask] = np.interp(
                    np.flatnonzero(mask),
                    np.flatnonzero(~mask),
                    x[~mask]
                )

    # cross-sectional normalization per time
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std

    return features

def build_obs_windows(raw_inputs, K):
    """
    Parameters
    ----------
    raw_inputs : np.ndarray
        Shape (N, T, F)
        raw_inputs[i, t, f] is feature f for asset i known at time t

    K : int
        Lookback window length

    Returns
    -------
    obs_windows : np.ndarray
        Shape (B, N, K, F), where B = T - K + 1
        obs_windows[b, i, k, f] corresponds to
        raw_inputs[i, b + k, f]

    time_index : np.ndarray
        time_index[b] = t corresponding to window end time
    """
    N, T, F = raw_inputs.shape
    B = T - K + 1

    obs_windows = np.zeros((B, N, K, F), dtype=np.float32)
    returns = np.zeros((N,K), dtype=np.float32)
    for b in range(B):
        # Window ends at time t = b + K - 1
        obs_windows[b] = raw_inputs[:, b:b+K, :-1]
        returns[b] = raw_inputs[:, b+ K, -1]

    time_index = np.arange(K - 1, T)

    return obs_windows, returns


>>>>>>> Stashed changes

    # Identify training mode (has forward_returns)
    is_train = "forward_returns" in df.columns

    # Drop high-null columns only during training
    if is_train:
        print("Training data Preprocess")
        high_null_cols = [c for c in df.columns if df[c].isnull().mean() > 0.5]
        drop_cols = high_null_cols  # Save for test data
    elif drop_cols is not None:
        # For test data, drop same columns as training
        df = df.drop(columns=drop_cols, errors='ignore')

    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            if len(df[col].mode()) > 0:
                df[col] = df[col].fillna(df[col].mode()[0])

    return df, drop_cols

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    targets = ['forward_returns', 'risk_free_rate']

    # Lag Features
    for col in targets:
        if col in df.columns:
            for lag in [1, 2, 3, 5, 10]:
                df[f'lag_{col}_{lag}'] = df[col].shift(lag)

    # Volatility Features 
    base_col = 'lag_forward_returns_1'
    if base_col not in df.columns and 'forward_returns' in df.columns:
        # lagging forward_returns 
        df[base_col] = df['forward_returns'].shift(1)

    
    if base_col not in df.columns:
        df[base_col] = 0.0

    df['vol_5d']  = df[base_col].rolling(5).std() # weekly
    df['vol_22d'] = df[base_col].rolling(22).std()  # monthly

    # Momentum Features
    df['mom_5d']  = df[base_col].rolling(5).mean()
    df['mom_22d'] = df[base_col].rolling(22).mean()

    return df


def temporal_zscore_normalize(X):
    """
    X shape: (samples, time_steps, num_features)
    Returns: normalized_X with same shape
    """
    # mean over time dimension (axis=1)
    mean = np.mean(X, axis=1, keepdims=True)           # shape (samples, 1, features)
    std  = np.std(X, axis=1, keepdims=True) + 1e-8     # avoid division by zero

    return (X - mean) / std

def create_dataset(X: pd.DataFrame, y: pd.DataFrame, time_step=20):
    '''
    Creating rolling look-back windows to create sequences of data.

    X: pd.DataFrame containing financial features (T, num_features)
    y: pd.DataFrame containing the next-day excess returns (T, 1)
    time_step: int that determines the lookback window length

    returns:
    X_out: np.array containing features within lookback wndows (T-time_step, time_step, num_features)
    y_out: np.array containing excess returns after lookback windows (T - time_step, 1)
    '''

    X_seq, Y_seq = [], []
    for i in range(time_step, len(X)):
        X_seq.append(X.iloc[i-time_step:i])
        Y_seq.append(y.iloc[i])

    
    X_out = temporal_zscore_normalize(X_seq)
    y_out = np.array(Y_seq)

    return X_out, y_out

# def create_dataset(X: pd.DataFrame, y: pd.DataFrame, time_step=20):
#     """
#     X: (T, num_features)
#     y: (T, 1)
#     """

#     X_seq, Y_seq = [], []
#     xscales = []; yscales = []
#     for i in range(time_step, len(X)):
#         # window of X
#         X_window = X.iloc[i-time_step:i].values  # (time_step, features)

#         # window of y (used ONLY for mean/std)
#         y_window = y.iloc[i-time_step:i].values  # (time_step, 1)
#         y_mean = y_window.mean()
#         y_std  = y_window.std() + 1e-8

#         # scale X window
#         scalerx = StandardScaler()
#         X_norm = scalerx.fit_transform(X_window[np.newaxis, ...][0])
#         xscales.append(scalerx)

#         # scale NEXT y (target), using y-window stats
#         scalery = StandardScaler()
#         y_next = y.iloc[i].values  # unscaled
#         y_next_norm = scalery.fit_transform(y_next)
#         yscales.append(scalery)

#         X_seq.append(X_norm)
#         Y_seq.append(y_next_norm)

#     return np.array(X_seq), np.array(Y_seq), xscales, yscales

