import tensorflow as tf
import numpy as np
from src.env import env as en
from src.utils import metrics as met

def non_grads(agent, loss, grads):
    none_grads = [v.name for v, g in zip(agent.trainable_variables, grads) if g is None]
    if none_grads:
        print(f"None gradients for: {none_grads}")
    else:
        print(f"All gradients flowing, loss: {loss.numpy():.6f}")

def rl_update(agent, buffer, 
              actor_optimizer, 
              critic_optimizer, 
              gamma=0.99, 
              sharpe_lambda=0.1,
              critic_updates = 5,
              grad_clip = 1.0):
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
    sharpe = met.sharpe_ratio_tf(rewards)
    T = tf.shape(obs)[0]
    B = tf.shape(obs)[1]
    
    # merge T and B dimensions
    obs_flat = tf.reshape(obs, (T * B, tf.shape(obs)[2], tf.shape(obs)[3], tf.shape(obs)[4]))  # (T*B, N, K, F)
    
    with tf.GradientTape(persistent = True) as tape:       
        pred_actions, pred_values = agent(obs_flat, training=True)  # (T*B, N), (T*B,)
        
        pred_actions = tf.reshape(pred_actions, (T, B, -1))  # (T, B, N)
        pred_values = tf.reshape(pred_values, (T, B))        # (T, B)

        # Actor Loss
        policy_loss = -tf.reduce_mean(tf.expand_dims(advantages, -1) * pred_actions)
        entropy = -tf.reduce_mean(pred_actions * tf.math.log(tf.nn.softmax(pred_actions, axis=-1) + 1e-8))
        actor_loss = policy_loss - 0.01 * entropy - sharpe_lambda * sharpe

        # Critic Loss
        value_loss = tf.reduce_mean((returns - pred_values) ** 2)
        
        # loss = policy_loss + 0.5 * value_loss - sharpe_lambda * sharpe - 0.01 * entropy
        # loss = policy_loss + 0.5 * value_loss + 0.05 * entropy
 
    # Compute and apply gradients
    actor_vars = (
    agent.srem.trainable_variables +
    agent.caan.trainable_variables +
    agent.portfolio_gen.trainable_variables
    )

    critic_vars = (
        agent.srem.trainable_variables +
        agent.caan.trainable_variables +
        agent.value_head.trainable_variables
    )

    # Updating critic K times before actor
    for _ in range(critic_updates):
        critic_grads = tape.gradient(value_loss, critic_vars)
        critic_grads = [tf.clip_by_norm(g, grad_clip) if g is not None else None for g in critic_grads]
        critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))
    
    none_grads = [v.name for v, g in zip(critic_vars, critic_grads) if g is None]
    if none_grads:
        print(f"(Critic) None gradients for: {none_grads}")
    else:
        print(f"(Critic) All gradients flowing, loss: {value_loss.numpy():.6f}")
    
    # Actor Gradients
    actor_grads = tape.gradient(actor_loss, actor_vars)
    actor_grads = [tf.clip_by_norm(g, grad_clip) if g is not None else None for g in actor_grads]
    actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))

    none_grads = [v.name for v, g in zip(actor_vars, actor_grads) if g is None]
    if none_grads:
        print(f"(Actor) None gradients for: {none_grads}")
    else:
        print(f"(Actor) All gradients flowing, loss: {actor_loss.numpy():.6f}")

    del tape

    return policy_loss.numpy(), actor_loss.numpy(), value_loss.numpy()

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

def train(agent, env, actor_optimizer, critic_optimizer, num_epochs, rollout_len, gamma=0.99, sharpe_lambda=0.1):
    buffer = en.RolloutBuffer()
    log = {key: [] for key in ["mean_reward","sharpe","max_drawdown","turnover","entropy","policy_loss","value_loss", "actor_loss"]}

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
        policy_loss, actor_loss, value_loss = rl_update(
            agent=agent,
            buffer=buffer,
            actor_optimizer= actor_optimizer,
            critic_optimizer=critic_optimizer,
            gamma=gamma,
            sharpe_lambda=sharpe_lambda
        )

        mean_reward = np.mean(buffer.rewards)
        metrics = met.compute_metrics(buffer)

        # Logging metrics
        log['mean_reward'].append(mean_reward)
        log['sharpe'].append(metrics['sharpe'])
        log['max_drawdown'].append(metrics['max_drawdown'])
        log['turnover'].append(metrics['turnover'])
        log['entropy'].append(metrics['entropy'])
        log['policy_loss'].append(policy_loss)
        log['value_loss'].append(value_loss)
        log['actor_loss'].append(value_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"Mean reward: {mean_reward:.6f} | "
            f"Sharpe: {metrics['sharpe']:.2f} | "
            f"MDD: {metrics['max_drawdown']:.2%} | "
            f"Turnover: {metrics['turnover']:.4f}"
        )

    return log


