import tensorflow as tf
import numpy as np


def rl_sharpe_loss(y_true, alloc):
    eps = 1e-8

    strat_returns = tf.squeeze(alloc) * tf.squeeze(y_true)

    mean_r = tf.reduce_mean(strat_returns)
    std_r = tf.math.reduce_std(strat_returns) + eps  # add eps for numerical stability

    sharpe = mean_r / std_r
    annualization_factor = tf.constant(np.sqrt(252), dtype=tf.float32)

    return -sharpe * annualization_factor


# A wrapper loss to use in Keras expects y_true and y_pred as tensors
def loss_wrapper(model, l1_lambda=1e-4):
    def loss_fn(y_true, y_pred):
        # Your current Sharpe loss
        sharpe_loss = rl_sharpe_loss(y_true, y_pred)
        
        # L1 penalty on model weights
        l1_penalty = tf.add_n([
            tf.reduce_sum(tf.abs(w))
            for w in model.trainable_weights
            if 'kernel' in w.name
        ])
        
        return sharpe_loss + l1_lambda * l1_penalty
    return loss_fn

# def loss_wrapper():
    
#     def loss_fn(y_true, y_pred):
#         # y_true: shape (batch, 1) next-day excess returns
#         # y_pred: predicted allocation in [0,2]
#         return rl_sharpe_loss(y_true, y_pred)
#     return loss_fn

# def sharpe_loss(y_true_excess_returns, predicted_alloc, risk_free_rate, market_excess_returns, solution, vol_cap):
#     """
#     y_true_excess_returns: shape (batch, 1) -> next day excess returns observed for that sample
#     predicted_alloc: shape (batch, 1) -> allocations in [0,2]
#     market_volatility: scalar or shape broadcastable -> e.g. historical market std
#     vol_cap: multiplier (1.2 = 120% of market vol) -> limit applied by competition
#     vol_penalty_coeff: coefficient to penalize exceeding vol cap

#     Loss is -mean(strategy_returns) / (std(strategy_returns) + eps)
#     plus a differentiable penalty when std(strategy_returns) > vol_cap * market_volatility
#     """
    
#     eps = 1e-8
#     # compute strategy returns per sample: allocation * next day * (1)  (risk-free already subtracted if using excess returns)
#     # y_true_excess_returns should already be *excess* returns (vs risk-free)
#     strat_returns = tf.squeeze(predicted_alloc, -1) * tf.squeeze(y_true_excess_returns, -1)  # shape (batch,)
#     mean_r = tf.reduce_mean(strat_returns)
#     std_r = tf.math.reduce_std(strat_returns)
#     trading_days_per_year = 252

#     # negative Sharpe (we minimize) - annualized
#     neg_sharpe = - ((mean_r - risk_free_rate) / (std_r + eps)) * np.sqrt(trading_days_per_year)
    
#     # volatility penalty (smooth hinge): penalize (std_r - vol_cap*market_volatility) if positive
#     excess_vol = tf.maximum(0.0, (std_r / vol_cap) - 1)
#     vol_penalty = 1 + excess_vol

#     # return penalty
#     market_excess_cumulative = tf.reduce_prod(1 + market_excess_returns)
#     market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
#     return_gap = tf.maximum(0.0,
#         (market_mean_excess_return - mean_r) * 100 * trading_days_per_year,
#     )
#     return_penalty = 1 + (return_gap**2) / 100

#     return neg_sharpe / (vol_penalty + return_penalty)