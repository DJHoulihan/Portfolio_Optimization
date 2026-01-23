import numpy as np

def max_drawdown(returns):
    returns = np.asarray(returns)
    equity_curve = np.cumprod(1.0 + returns)

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak

    return np.max(drawdown)

def sharpe_ratio(returns, eps=1e-8):
    returns = np.asarray(returns)
    mean = returns.mean()
    std = returns.std() + eps
    return np.sqrt(252) * mean / std

def turnover(weights):
    """
    weights: (T, B, N)
    """
    weights = np.asarray(weights)

    diffs = np.abs(weights[1:] - weights[:-1])
    step_turnover = np.sum(diffs, axis=-1)  # (T-1, B)

    return np.mean(step_turnover)


def compute_metrics(buffer):
    rewards = np.stack(buffer.rewards)     # (T, B)
    actions = np.stack(buffer.actions)     # (T, B, N)

    # Portfolio returns per env
    mean_returns = rewards.mean(axis=1)    # (T,)

    metrics = {
        "sharpe": sharpe_ratio(mean_returns),
        "max_drawdown": max_drawdown(mean_returns),
        "turnover": turnover(actions),
        "mean_return": mean_returns.mean(),
        "volatility": mean_returns.std(),
    }

    return metrics
