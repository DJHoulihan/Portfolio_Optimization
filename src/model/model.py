import tensorflow as tf
from tensorflow.keras import Model
from src.model import portfolio_generator as pg
from src.model import srem as sr
from src.model import caan as cn
from src.env import env as en
from src.utils import metrics as met
import numpy as np


class Actor(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.srem = sr.SREM(config)
        self.caan = cn.CAAN(config)
        self.log_std = None  # will initialize later

        self.mu_head = None  # will initialize in build()

    def build(self, input_shape):
        # input_shape: (batch_size, N, K, F)
        num_assets = input_shape[1]  # N dimension = number of assets
        self.mu_head = tf.keras.layers.Dense(num_assets)
        self.log_std = tf.Variable(tf.zeros(num_assets), trainable=True)
        super().build(input_shape)

    def call(self, inputs, training=None, mask = None):
        z = self.srem(inputs, training=training)
        h = self.caan(z, training=training)
        features = tf.reduce_mean(h, axis=1)
        mu = self.mu_head(features)
        std = tf.exp(self.log_std)
        return mu, std
    
    def sample_action(self, inputs):

        mu, std = self(inputs, training=True)

        eps = tf.random.normal(tf.shape(mu))
        raw_action = mu + std * eps

        # convert to valid portfolio weights
        weights = tf.nn.softmax(raw_action, axis=-1)

        # log prob of Gaussian BEFORE softmax
        log_prob = -0.5 * (
            ((raw_action - mu) / std)**2 +
            2 * tf.math.log(std) +
            tf.math.log(2 * np.pi)
        )

        log_prob = tf.reduce_sum(log_prob, axis=-1)

        return weights, log_prob

class Critic(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.srem = sr.SREM(config)
        self.caan = cn.CAAN(config)
        self.value_head = None  # initialize in build()

    def build(self, input_shape):
        # input_shape: (batch_size, N, K, F)
        self.value_head = tf.keras.layers.Dense(1)
        super().build(input_shape)

    def call(self, inputs, training=False, mask = None):
        z = self.srem(inputs, training=training)
        h = self.caan(z, training=training)
        value = self.value_head(tf.reduce_mean(h, axis=1))
        return tf.squeeze(value, axis=-1)
    
class HouliPortfolio(Model):

    def __init__(self, config,
                 actor_lr=1e-4,
                 critic_lr=1e-3):

        super().__init__()

        self.config = config
        self.actor = Actor(config)
        self.critic = Critic(config)

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    
    def collect_rollout(self, env, buffer, rollout_len, training):

        obs = env.reset()

        for _ in range(rollout_len):

            obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)

            actions, log_probs = self.actor.sample_action(obs_tf)
            values = self.critic(obs_tf, training = training)

            next_obs, rewards, dones, _ = env.step(actions.numpy())

            buffer.add(
                obs=obs,
                action=actions.numpy(),
                reward=rewards,
                value=values.numpy(),
                log_prob=log_probs.numpy(),
                done=dones
            )

            obs = next_obs
    
    def _update(self, buffer):

        obs = tf.convert_to_tensor(buffer.obs, dtype=tf.float32)
        rewards = tf.convert_to_tensor(buffer.rewards, dtype=tf.float32)
        values = tf.convert_to_tensor(buffer.values, dtype=tf.float32)

        next_values = tf.concat([values[1:], tf.zeros_like(values[-1:])], axis=0)

        returns = rewards + self.config.training.gamma * next_values
        advantages = returns - values

        advantages = tf.stop_gradient((advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8))

        # sharpe = met.sharpe_ratio_tf(rewards)

        T = tf.shape(obs)[0]
        B = tf.shape(obs)[1]

        obs_flat = tf.reshape(obs, (T * B, tf.shape(obs)[2], tf.shape(obs)[3], tf.shape(obs)[4]))
        actions_flat = tf.reshape(tf.convert_to_tensor(buffer.actions, dtype=tf.float32), (T*B, -1))

        # =====================
        # Critic Updates
        # =====================
        for _ in range(self.config.training.critic_updates):

            with tf.GradientTape() as tape:

                pred_values = self.critic(obs_flat, training=True)
                pred_values = tf.reshape(pred_values, (T, B))

                value_loss = tf.reduce_mean((returns - pred_values) ** 2)

            grads = tape.gradient(
                value_loss,
                self.critic.trainable_variables
            )

            grads = [
                tf.clip_by_norm(g, self.config.training.grad_clip) if g is not None else None
                for g in grads
            ]

            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        # =====================
        # Actor Update
        # =====================
        with tf.GradientTape() as tape:
            # Recompute mu and std
            mu, std = self.actor(obs_flat, training=True)
            log_probs = -0.5 * (((actions_flat - mu)/std)**2 + 2*tf.math.log(std) + tf.math.log(2*np.pi))
            log_probs = tf.reduce_sum(log_probs, axis=-1)
            log_probs = tf.reshape(log_probs, (T,B))

            policy_loss = -tf.reduce_mean(advantages * log_probs)

        grads = tape.gradient(
            policy_loss,
            self.actor.trainable_variables
        )

        grads = [
            tf.clip_by_norm(g, self.config.training.grad_clip) if g is not None else None
            for g in grads
        ]

        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        return policy_loss.numpy(), policy_loss.numpy(), value_loss.numpy()

    def learn(self, env):

        buffer = en.RolloutBuffer()

        log = {key: [] for key in [
            "mean_reward",
            "sharpe",
            "max_drawdown",
            "turnover",
            "policy_loss",
            "value_loss",
            "actor_loss"
        ]}

        for epoch in range(self.config.training.epochs):

            buffer.clear()

            self.collect_rollout(env, buffer, self.config.training.rollout_len, training = True)

            policy_loss, actor_loss, value_loss = self._update(buffer)

            mean_reward = np.mean(buffer.rewards)
            metrics = met.compute_metrics(buffer)

            log['mean_reward'].append(mean_reward)
            log['sharpe'].append(metrics['sharpe'])
            log['max_drawdown'].append(metrics['max_drawdown'])
            log['turnover'].append(metrics['turnover'])
            log['policy_loss'].append(policy_loss)
            log['value_loss'].append(value_loss)
            log['actor_loss'].append(actor_loss)

            print(
                f"Epoch {epoch:03d} | "
                f"Mean reward: {mean_reward:.6f} | "
                f"Sharpe: {metrics['sharpe']:.2f} | "
                f"MDD: {metrics['max_drawdown']:.2%} | "
                f"Turnover: {metrics['turnover']:.4f}"
            )

        return log