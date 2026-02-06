import tensorflow as tf
from tensorflow.keras import Model
from src.model import portfolio_generator as pg
from src.model import srem as sr
from src.model import caan as cn

class PortfolioAgentCritic(Model):
    def __init__(self, config):
        super().__init__()
        self.srem = sr.SREM(config)        # outputs (B*N, d)
        self.caan = cn.CAAN(config)        # outputs (B, N,
        self.portfolio_gen = pg.PortfolioGenerator(config)
        self.value_head = tf.keras.layers.Dense(1)
        self.d = config.srem.embed_dim 
        self.log_sigma = tf.Variable(0.0)

    def call(self, x, mask = None, training=False):
        """
        x: (B, N, K, F)
        returns: weights (B, N)
        """

        B, N, K, F = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # Flatten assets for SREM
        x_flat = tf.reshape(x, (B * N, K, F))  # (B*N, K, F)

        # SREM
        z_flat = self.srem(x_flat, training=training)  # (B*N, d)

        # Reshape back
        z = tf.reshape(z_flat, (B, N, self.d))  # (B, N, d)

        # CAAN
        h = self.caan(z, training=training)  # (B, N, d)

        # Portfolio weights (Agent)
        actions, _ = self.portfolio_gen(h, mask)

        # Value head (Critic)
        value = self.value_head(tf.reduce_mean(h, axis=1))

        return actions, tf.squeeze(value, axis = -1)

