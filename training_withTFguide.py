import tensorflow as tf
import numpy as np
from src.model import portfolio_generator as pg
from src.env import env as en
from src.env import agent as ag
from src.preprocess import dataprep as dp
from src.config import config as cf
from src.preprocess import datapull as dpl
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

# Wrap Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""
  env = en.PortfolioEnv

  state, reward, done, _ = env.step(action)
  return (state.astype(np.float32),
          np.array(reward, np.int32),
          np.array(done, np.int32))

def collect_rollout(env, agent, buffer, rollout_len):
    """
    Collects a fixed-length rollout from a vectorized PortfolioEnv.

    env.reset() returns (B, N, K, F)
    """
    obs = env.reset()

    for _ in range(rollout_len):
        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)

        # Forward pass
        actions, values = agent(obs_tf, training=False)  # (B, N), (B,)

        # Step environment (NumPy)
        next_obs, rewards, dones, _ = env.step(actions.numpy())

        buffer.add(
            obs=obs,
            action=actions.numpy(),
            reward=rewards,
            value=values.numpy(),
            log_prob=None,   # deterministic policy
            done=dones
        )

        obs = next_obs
# def main():
#     """
#     Train PortfolioAgent using Actor-Critic RL structure.
    
#     Parameters
#     ----------
#     agent : PortfolioAgent
#         The agent to train
#     obs_windows : np.ndarray
#         Market data: shape (T, N, K, F)
#     returns : np.ndarray
#         Realized returns: shape (T, N)
#     optimizer : tf.keras.optimizers.Optimizer
#         Optimizer for agent parameters
#     input_window : int
#         Number of months for agent input (state)
#     reward_window : int
#         Number of months to compute Sharpe ratio reward
#     epochs : int
#         Number of passes over the training data
#     gamma : float
#         Discount factor
#     sharpe_lambda : float
#         Weight for Sharpe ratio bonus
#     """
#     # Set up configurations
#     config = cf.load_config()
#     tconfig = config.training

#     # Pull data from alphavantage
#     datapull = dpl.AlphaVantageData(config.api.api_key)
#     data = datapull.get_daily_data(list(config.data.symbols))

#     T = data.shape[0]
        
#     for epoch in range(tconfig.epochs):
#         # Track which months have been used in this epoch
#         months_used = set()
#         # Months eligible for random draw
#         months_available = list(range(tconfig.input_window-1, T - tconfig.reward_window))
#         np.random.shuffle(months_available)

#         for t_drawn in months_available:
#             if t_drawn in months_used:
#                 continue

#             # Input for agent: previous `input_window` months including t_drawn
#             obs_input = obs_windows[t_drawn - input_window + 1 : t_drawn + 1]  # (input_window, N, K, F)
            
#             # Reward calculation: next `reward_window` months
#             returns_window = returns[t_drawn + 1 : t_drawn + 1 + reward_window]  # (reward_window, N)

#             # Create environment for this episode
#             env = en.PortfolioEnv(obs_input, returns_window, num_envs=1)
            
#             # Initialize rollout buffer
#             buffer = en.RolloutBuffer()
            
#             # Reset environment
#             obs = env.reset()  # (B=1, N, K, F)
#             done = False
            
#             while not done:
#                 # Agent selects action
#                 action, value = agent(tf.expand_dims(obs, axis=0), training=True)
#                 action = tf.squeeze(action, axis=0).numpy()  # (N,)
#                 value = tf.squeeze(value, axis=0).numpy()    # scalar

#                 # Step environment
#                 next_obs, reward, dones, _ = env.step(np.expand_dims(action, axis=0))
#                 done = dones[0]
                
#                 # Log probability placeholder (we're using deterministic continuous actions here)
#                 log_prob = 0.0
                
#                 # Add to buffer
#                 buffer.add(obs, action, reward[0], value, log_prob, done)
                
#                 # Move to next observation
#                 obs = next_obs[0]

#             # RL update
#             en.rl_update(agent, buffer, optimizer, gamma=gamma, sharpe_lambda=sharpe_lambda)
            
#             months_used.add(t_drawn)

#         print(f"Epoch {epoch+1}/{epochs} complete")


# if __name__ == "__main__":
#     main()




# def main(data):

#     config = cf.load_config()
#     obs_windows, returns = dp.create_windows(data, config.lookback)
#     (train_obs, train_ret), (val_obs, val_ret), _ = dp.train_val_test_split(obs_windows, returns)

#     env = en.PortfolioEnv(train_obs, train_ret, config.num_envs)
#     agent = ag.PortfolioAgent(config)
#     optimizer = tf.keras.optimizers.Adam(learning_rate = config.srem.learning_rate)

#     buffer = en.RolloutBuffer()

#     for epoch in range(config.epochs):
#         obs = env.reset()
#         done = np.zeros(config.num_envs, dtype=bool)

#         for step in range(config.steps_per_epoch):
#             actions, values = agent(obs, training=False)
#             actions = actions.numpy()
#             values = values.numpy()

#             # log probs placeholder (you need a proper distribution if you want log probs)
#             log_probs = np.zeros(config.num_envs)

#             next_obs, rewards, dones, _ = env.step(actions)

#             for i in range(config.num_envs):
#                 buffer.add(obs[i], actions[i], rewards[i], values[i], log_probs[i], dones[i])

#             obs = next_obs

#             if np.all(dones):
#                 break

#         en.rl_update(agent, buffer, optimizer)
#         buffer.clear()

#         print(f"Epoch {epoch} complete.")

