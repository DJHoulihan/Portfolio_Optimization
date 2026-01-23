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


