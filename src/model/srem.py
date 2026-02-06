import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
import src.model.transformer as tm
import logging


class SREM(Layer):
    """
    Shared Representation Extraction Module (taken from AlphaPortfolio)
    """

    def __init__(self, config, **kwargs):
        super(SREM, self).__init__(**kwargs)
        # logging.info(f"Initializing SREM with d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        
        # hyperparameters 
        self.lookback = config.training.lookback     # K
        self.embed_dim = config.srem.embed_dim   # d

        # hyperparameters for transformer
        self.num_layers = config.srem.num_layers
        self.num_heads = config.srem.num_heads
        self.ff_dim = config.srem.ff_dim
        self.dropout_rate = config.srem.dropout_rate

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

    def call(self, x, training=False):
        """
        inputs:  (Batch (B), Assets (N), Lookback Window (K), Features (F))
        returns: (B, N, K, embed dim (d))
        """

        B = tf.shape(x)[0] # batch size
        N = tf.shape(x)[1] # number of assets
        K = tf.shape(x)[2]
        F = tf.shape(x)[3]
        d = self.embed_dim

        if K != self.lookback:
            ValueError('Unexpected number of assets.')

        if d != self.embed_dim:
            ValueError('Unexpected or incorrect embedding dimension.')

        # flatten asset dimension 
        # Dimension: (B*N, K, F)
        x = tf.reshape(x, (B * N, K, F)) 

        # embed features
        x = self.input_projection(x)

        # positional encoding
        x = self.positional_encoding(x)

        # temporal self-attention (shared weights)
        z = self.temporal_encoder(x, training=training)

        # normalization
        z = self.dropout(z, training=training)

        # ---- restore asset dimension
        z = tf.reshape(z, (B, N, K, d))

        # aggregate via attention scores -> (B, N, d)
        att_scores = self.attn_weight(z)
        att_weights = tf.nn.softmax(att_scores, axis = 2)
        r = tf.reduce_sum(z * att_weights, axis = 2)

        return r
