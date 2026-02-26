import tensorflow as tf
import src.env.env as en
import src.env.agent as ag
import src.train.trainingfuncs as tfu
import src.config.config as cf
import src.preprocess.datapull as dp
import os
import warnings
import json
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import src.preprocess.dataprep as dpr

def main(config):
    # config = cf.load_config()

    # ----- Data -----
    datahandler = dp.DataHandler(config)
    obs_windows, returns = datahandler.load_data()

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
    log = tfu.train(
        agent=agent,
        env=env,
        optimizer=optimizer,
        num_epochs=config.training.epochs, 
        rollout_len=config.training.rollout_len,
        gamma=config.training.gamma,
        sharpe_lambda=config.training.sharpe_lambda,
    )

    # Saving metric logs to json file
    os.makedirs("logs", exist_ok=True)

    with open("logs/training_log.txt", "w") as f:
        for epoch in range(len(log['mean_reward'])):
            line = (
                f"Epoch {epoch:03d} | "
                f"Mean reward: {log['mean_reward'][epoch]:.6f} | "
                f"Sharpe: {log['sharpe'][epoch]:.2f} | "
                f"MDD: {log['max_drawdown'][epoch]:.2%} | "
                f"Turnover: {log['turnover'][epoch]:.4f} | "
                f"Entropy: {log['entropy'][epoch]:.4f} | "
                f"Policy loss: {log['policy_loss'][epoch]:.6f} | "
                f"Value loss: {log['value_loss'][epoch]:.6f}\n"
            )
            f.write(line)

    print("Logs saved to logs/training_log.txt")

    # Save model weights
    os.makedirs("checkpoints", exist_ok=True)
    agent.save_weights("checkpoints/agent.weights.h5")
    print("Training logs and model weights saved.")

    # # Save model
    # model_file = os.path.join("checkpoints", "full_agent.h5")
    # agent.save(model_file)
    # print(f"Full model saved to {model_file}")


if __name__ == "__main__":
    config = cf.load_config()
    main(config)
