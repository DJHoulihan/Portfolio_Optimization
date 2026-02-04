import tensorflow as tf
import src.env.env as en
import src.env.agent as ag
import src.train.trainingfuncs as tfu
import src.config.config as cf
import src.preprocess.datapull as dp
# import src.preprocess.dataprep as dpr

def main(config):
    # ----- Data -----
    datahandler = dp.DataHandler(config)
    obs_windows, returns = datahandler.load_data(config.data)

    # ----- Environment -----
    env = en.PortfolioEnv(
        obs_windows=obs_windows,
        returns=returns,
        num_envs=config.env.num_envs
    )

    # ----- Agent -----
    agent = ag.PortfolioAgentCritic(config)

    # ----- Optimizer -----
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.training.learning_rate
    )

    # ----- Train -----
    tfu.train(
        agent=agent,
        env=env,
        optimizer=optimizer,
        num_epochs=config.training.num_epochs,
        rollout_len=config.training.rollout_len,
        gamma=config.training.gamma,
        sharpe_lambda=config.training.sharpe_lambda,
    )

if __name__ == "__main__":
    config = cf.load_config()
    main(config)
