import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Model
from ..model import transformer as tm
from ..model import portfolio_generator as pg
from ..model import srem as sr
from ..model import caan as cn

class PortfolioAgent(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.srem = sr.SREM(config)        # outputs (B*N, d)
        self.caan = cn.CAAN(config)        # outputs (B, N,
        self.portfolio_gen = pg.PortfolioGenerator(config)
        self.score_head = tf.keras.layers.Dense(1)
        self.value_head = tf.keras.layers.Dense(1)
        self.d = config.embed_dim
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

        # Portfolio weights
        actions, _ = self.portfolio_gen(h, mask)

        # Value head
        value = self.value_head(tf.reduce_mean(h, axis=1))

        return actions, tf.squeeze(value, axis = -1)

