import numpy as np

class PortfolioEnv:
    def __init__(self, obs_windows, returns, num_envs = 8, cost_rate=0.001):
        self.obs_windows = obs_windows
        self.returns = returns.copy()
        self.num_envs = num_envs
        self.cost_rate = cost_rate

        self.T = obs_windows.shape[0]
        self.N = returns.shape[1]

        self.reset()

    def reset(self):
        self.t = np.random.randint(0, self.T - 1, size=self.num_envs)
        self.prev_actions = np.zeros((self.num_envs, self.N)) / self.N
        return self.obs_windows[self.t]
    
    def step(self, actions):
        """
        Vectorized environment step.
        actions: (B, N)
        returns: (T, N)
        """

        # Current time indices (B,)
        t = self.t.copy()
        t = np.minimum(t, self.T-1)

        # Gather returns for each env at its current timestep
        r = self.returns[t]            # (B, N)

        # Portfolio reward
        rewards = np.einsum("bn,bn->b", actions, r)
        # Transaction costs (turnover)
        turnover = np.sum(np.abs(actions - self.prev_actions), axis=1)
        # rewards -= self.cost_rate * turnover

        # Update state
        self.prev_actions = actions.copy()
        self.t += 1
        # Done flags
        dones = self.t >= (self.T - 1)
        

        # Next observations
        next_obs = self.obs_windows[self.t.clip(max=self.T - 1)]

        # Zero-out observations for finished envs
        next_obs[dones] = 0.0

        info = {
        "rewards": rewards,
        "turnover": turnover,
        "timestep": self.t.copy()
        }

        return next_obs, rewards, dones, info

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.info = []

    def add(self, obs, action, reward, value, log_prob, done, info=None):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.info.append(info if info is not None else {})

    def clear(self):
        self.__init__()
    
    def get_raw_returns(self):
        """
        Concatenate all raw portfolio returns collected in the rollout.
        Useful for computing Sharpe, MDD, etc.
        """
        raw_returns_list = []
        for step_info in self.info:
            if "portfolio_raw_returns" in step_info:
                raw_returns_list.append(step_info["portfolio_raw_returns"])
        if raw_returns_list:
            return np.concatenate(raw_returns_list, axis=0)
        else:
            return np.array([])

    def get_turnover(self):
        """
        Concatenate all turnover values collected in the rollout.
        """
        turnover_list = []
        for step_info in self.info:
            if "turnover" in step_info:
                turnover_list.append(step_info["turnover"])
        if turnover_list:
            return np.concatenate(turnover_list, axis=0)
        else:
            return np.array([])

    # def _normalize(self, action):
    #     action = np.clip(action, 1e-6, 1.0)
    #     return action / action.sum() 

    # def step(self, actions):
    #     rewards = np.zeros(self.num_envs)
    #     dones = np.zeros(self.num_envs, dtype=bool)
    #     next_obs = []

    #     for i in range(self.num_envs):
    #         t = self.t[i]

    #         # action = self._normalize(actions[i])
    #         action = actions[i]
    #         r = self.returns[t]

    #         reward = action @ r

    #         turnover = np.sum(np.abs(action - self.prev_actions[i]))
    #         reward -= self.cost_rate * turnover

    #         rewards[i] = reward
    #         self.prev_actions[i] = action.copy()

    #         self.t[i] += 1
    #         dones[i] = self.t[i] >= self.T - 1

    #         if dones[i]:
    #             next_obs.append(np.zeros_like(self.obs_windows[0]))
    #         else:
    #             next_obs.append(self.obs_windows[self.t[i]])

    #     return np.stack(next_obs), rewards, dones, {}
