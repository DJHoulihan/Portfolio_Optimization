import numpy as np

class PortfolioEnv:
    def __init__(self, obs_windows, returns, num_envs = 8, cost_rate=0.0):
        self.obs_windows = obs_windows
        self.returns = returns
        self.num_envs = num_envs
        self.cost_rate = cost_rate

        self.T = obs_windows.shape[0]
        self.N = returns.shape[1]

        self.reset()

    def reset(self):
        self.t = np.arange(self.num_envs)
        self.prev_actions = np.ones((self.num_envs, self.N)) / self.N
        return self.obs_windows[self.t]

    def _normalize(self, action):
        action = np.clip(action, 1e-6, 1.0)
        return action / action.sum() 

    def step(self, actions):
        rewards = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)
        next_obs = []

        for i in range(self.num_envs):
            t = self.t[i]

            action = self._normalize(actions[i])
            r = self.returns[t]

            reward = action @ r

            turnover = np.sum(np.abs(action - self.prev_actions[i]))
            reward -= self.cost_rate * turnover

            rewards[i] = reward
            self.prev_actions[i] = action

            self.t[i] += 1
            dones[i] = self.t[i] >= self.T - 1

            if dones[i]:
                next_obs.append(np.zeros_like(self.obs_windows[0]))
            else:
                next_obs.append(self.obs_windows[self.t[i]])

        return np.stack(next_obs), rewards, dones, {}


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


