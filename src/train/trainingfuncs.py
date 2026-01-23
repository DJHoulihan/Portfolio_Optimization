import tensorflow as tf
import numpy as np
from src.env import env as en
from src.utils import metrics as met

def rl_update(agent, buffer, optimizer, gamma=0.99, sharpe_lambda=1.0):
    """
    Actor-Critic RL update for PortfolioAgent.
    Policy (actor) loss encourages actions proportional to advantages.
    Value (critic) loss minimizes TD(0) error.
    Sharpe ratio acts as a global reward bonus.
    
    Parameters
    ----------
    agent : tf.keras.Model
        Your PortfolioAgent
    buffer : RolloutBuffer
        Contains obs, actions, rewards, values
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer for agent parameters
    gamma : float
        Discount factor
    sharpe_lambda : float
        Weight for Sharpe ratio bonus
    """
    
    # Convert buffer lists to tensors
    obs = tf.convert_to_tensor(buffer.obs, dtype=tf.float32)          # (T, B, N, K, F)
    actions = tf.convert_to_tensor(buffer.actions, dtype=tf.float32)  # (T, B, N)
    rewards = tf.convert_to_tensor(buffer.rewards, dtype=tf.float32)  # (T, B)
    values = tf.convert_to_tensor(buffer.values, dtype=tf.float32)    # (T, B)

    # Compute simple TD(0) returns and advantages
    next_values = tf.concat([values[1:], tf.zeros_like(values[-1:])], axis=0)
    returns = rewards + gamma * next_values
    advantages = returns - values
    
    # Normalize advantages for stability
    advantages = tf.stop_gradient(
        (advantages - tf.reduce_mean(advantages)) /
        (tf.math.reduce_std(advantages) + 1e-8)
    )

    # Compute trajectory Sharpe ratio (annualized)
    sharpe = met.sharpe_ratio(rewards)
    
    with tf.GradientTape() as tape:
        # Forward pass through agent
        pred_actions, pred_values = agent(obs, training=True)  # (B, N), (B,)

        # Policy loss (actor) — encourage actions proportional to advantage
        # Here we assume higher action value is better if advantage > 0
        policy_loss = -tf.reduce_mean(advantages * rewards)

        # Value loss (critic) — TD error
        value_loss = tf.reduce_mean((returns - tf.squeeze(pred_values)) ** 2)

        # Total loss with Sharpe bonus
        loss = policy_loss + 0.5 * value_loss - sharpe_lambda * sharpe

    # Compute and apply gradients
    grads = tape.gradient(loss, agent.trainable_variables)
    optimizer.apply_gradients(zip(grads, agent.trainable_variables))

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

def train(agent, env, optimizer, num_epochs, rollout_len, gamma=0.99, sharpe_lambda=0.1):
    buffer = en.RolloutBuffer()

    for epoch in range(num_epochs):
        buffer.clear()

        # Collect experience
        collect_rollout(
            env=env,
            agent=agent,
            buffer=buffer,
            rollout_len=rollout_len
        )

        # Update actor–critic
        rl_update(
            agent=agent,
            buffer=buffer,
            optimizer=optimizer,
            gamma=gamma,
            sharpe_lambda=sharpe_lambda
        )

        mean_reward = np.mean(buffer.rewards)

        metrics = met.compute_metrics(buffer)

        print(
            f"Epoch {epoch:04d} | "
            f"Mean reward: {mean_reward:.6f}"
            f"Return: {metrics['mean_return']:.5f} | "
            f"Sharpe: {metrics['sharpe']:.2f} | "
            f"MDD: {metrics['max_drawdown']:.2%} | "
            f"Turnover: {metrics['turnover']:.2f}"
        )


