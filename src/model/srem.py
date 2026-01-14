import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
import transformer as tm
import os
import sys
import logging

# Add project root to sys.path (two levels up from current file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from config.config import get_config

class SREM(Layer):
    """
    Shared Representation Extraction Module (taken from AlphaPortfolio)
    """

    def __init__(self, config, **kwargs):
        super(SREM, self).__init__(**kwargs)
        # logging.info(f"Initializing SREM with d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        # hyperparameters
        self.lookback = int(config['SREM']['lookback'])       # K
        self.num_features = int(config['SREM']['num_features']) # F
        self.embed_dim = int(config['SREM']['embed_dim'])   # d

        # parameters for transformer
        self.num_layers = int(config['SREM']['num_layers'])
        self.num_heads = int(config['SREM']['num_heads'])
        self.ff_dim = int(config['SREM']['ff_dim'])
        self.dropout_rate = float(config['SREM']['dropout_rate'])

        # feature embedding
        self.input_projection = Dense(self.embed_dim)

        # positional encoding 
        self.positional_encoding = tm.PositionalEncoding(self.embed_dim)

        # shared transformer encoder
        self.temporal_encoder = tm.TransformerEncoder(
            num_layers= self.num_layers,
            embed_dim= self.embed_dim,
            num_heads= self.num_heads,
            ff_dim= self.ff_dim,
            rate= self.dropout_rate
        )

        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(self.dropout_rate)
        self.att_weight = Dense(1)

    def call(self, inputs, training=False):
        """
        inputs:  (Batch (B), Assets (N), Lookback Window (K), Features (F))
        returns: (B, N, K, embed dim (d))
        """

        B = tf.shape(inputs)[0] # batch size
        N = tf.shape(inputs)[1] # number of assets

        # flatten asset dimension 
        # Dimension: (B*N, K, F)
        x = tf.reshape(inputs, (B * N, self.lookback, self.num_features)) 

        # embed features
        x = self.input_projection(x)

        # positional encoding
        x = self.positional_encoding(x)

        # temporal self-attention (shared weights)
        z = self.temporal_encoder(x, training=training)

        # normalization
        z = self.dropout(z, training=training)

        # ---- restore asset dimension
        z = tf.reshape(x, (B, N, self.lookback, self.embed_dim))

        # aggregate via attention scores -> (B, N, d)
        att_scores = self.attn_weight(z)
        att_weights = tf.nn.softmax(att_scores, axis = 2)
        r = tf.reduce_sum(z * att_weights, axis = 2)

        return r
