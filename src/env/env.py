# import gym
import tensorflow as tf
import numpy as np

class PortfolioEnv:
    def __init__(self, obs_windows, returns, num_envs):
        """
        obs_windows: (T, N, K, F)
        returns: (T, N)
        """
        self.obs_windows = obs_windows
        self.returns = returns
        self.num_envs = num_envs
        self.T = obs_windows.shape[0]
        self.t = np.zeros(num_envs, dtype=int)

    def reset(self):
        self.t = np.zeros(self.num_envs, dtype=int)
        return self.obs_windows[self.t]  # (num_envs, N, K, F)

    def step(self, actions):
        """
        actions: (num_envs, N)
        """
        rewards = []
        next_obs = []
        dones = []

        for i in range(self.num_envs):
            t = self.t[i]
            r = self.returns[t]                # (N,)
            reward = np.sum(actions[i] * r)    # scalar
            rewards.append(reward)

            self.t[i] += 1
            done = self.t[i] >= self.T
            dones.append(done)

            if not done:
                next_obs.append(self.obs_windows[self.t[i]])
            else:
                next_obs.append(np.zeros_like(self.obs_windows[0]))

        return np.stack(next_obs), np.array(rewards), np.array(dones), {}


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, obs, action, reward, value, log_prob, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.__init__()


def sharpe_ratio(rewards): # annualized sharpe ratio
    r = tf.convert_to_tensor(rewards, dtype=tf.float32)
    return tf.reduce_mean(r) / (tf.math.reduce_std(r) + 1e-8) * tf.constant(np.sqrt(252), dtype=tf.float32)

def rl_update(agent, buffer, optimizer, clip_ratio=0.2, gamma=0.99, lam=0.95, sharpe_lambda=1.0):
    # Convert lists to tensors
    obs = tf.convert_to_tensor(buffer.obs, dtype=tf.float32)
    actions = tf.convert_to_tensor(buffer.actions, dtype=tf.float32)
    rewards = tf.convert_to_tensor(buffer.rewards, dtype=tf.float32)
    values = tf.convert_to_tensor(buffer.values, dtype=tf.float32)
    log_probs = tf.convert_to_tensor(buffer.log_probs, dtype=tf.float32)

    # Compute advantages (simple version)
    returns = rewards + gamma * tf.concat([values[1:], tf.zeros(1)], axis=0)
    advantages = returns - values

    sharpe = sharpe_ratio(rewards)

    with tf.GradientTape() as tape:
        pred_actions, pred_values = agent(obs, training=True)

        # Policy loss (clip)
        ratio = tf.exp(log_probs - tf.stop_gradient(log_probs))
        clipped = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))

        # Value loss
        value_loss = tf.reduce_mean((returns - pred_values) ** 2)

        # Total loss with Sharpe bonus
        loss = policy_loss + 0.5 * value_loss - sharpe_lambda * sharpe

    grads = tape.gradient(loss, agent.trainable_variables)
    optimizer.apply_gradients(zip(grads, agent.trainable_variables))