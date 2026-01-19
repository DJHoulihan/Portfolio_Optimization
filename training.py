import tensorflow as tf
import numpy as np
from src.model import portfolio_generator as pg
from src.env import env as en
from src.env import agent as ag
from src.preprocess import dataprep as dp
from src.config import config as cf

def main(data):

    config = cf.load_config()
    obs_windows, returns = dp.create_windows(data, config.lookback)
    (train_obs, train_ret), (val_obs, val_ret), _ = dp.train_val_test_split(obs_windows, returns)

    env = en.PortfolioEnv(train_obs, train_ret, config.num_envs)
    agent = ag.PortfolioAgent(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate = config.learning_rate)

    buffer = en.RolloutBuffer()

    for epoch in range(config.epochs):
        obs = env.reset()
        done = np.zeros(config.num_envs, dtype=bool)

        for step in range(config.steps_per_epoch):
            actions, values = agent(obs, training=False)
            actions = actions.numpy()
            values = values.numpy()

            # log probs placeholder (you need a proper distribution if you want log probs)
            log_probs = np.zeros(config.num_envs)

            next_obs, rewards, dones, _ = env.step(actions)

            for i in range(config.num_envs):
                buffer.add(obs[i], actions[i], rewards[i], values[i], log_probs[i], dones[i])

            obs = next_obs

            if np.all(dones):
                break

        en.rl_update(agent, buffer, optimizer)
        buffer.clear()

        print(f"Epoch {epoch} complete.")

if __name__ == "__main__":
    main()