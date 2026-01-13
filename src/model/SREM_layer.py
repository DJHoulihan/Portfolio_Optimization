import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
import TransformerModel as tm

class SREM(Layer):
    """
    Shared Representation Extraction Module (AlphaPortfolio)
    """

    def __init__(self,
                 lookback,
                 num_features,
                 embed_dim,
                 num_heads,
                 ff_dim,
                 num_layers,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)

        # hyperparameters
        self.lookback = lookback          # K
        self.num_features = num_features # F
        self.embed_dim = embed_dim        # d

        # feature embedding
        self.input_projection = Dense(embed_dim)

        # positional encoding 
        self.positional_encoding = tm.PositionalEncoding(embed_dim)

        # shared transformer encoder
        self.temporal_encoder = tm.TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            rate=dropout_rate
        )

        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        """
        inputs:  (Batch (B), Assets (N), Lookback Window (K), Features (F))
        returns: (B, N, K, embed dim (d))
        """

        B = tf.shape(inputs)[0] # batch size
        N = tf.shape(inputs)[1] # number of assets

        # flatten asset dimension 
        # Dimension: (B*N, K, F)
        x = tf.reshape(
            inputs,
            (B * N, self.lookback, self.num_features)
        ) 

        # embed features
        x = self.input_projection(x)

        # positional encoding
        x = self.positional_encoding(x)

        # temporal self-attention (shared weights)
        x = self.temporal_encoder(x, training=training)

        # normalization and regularization
        x = self.norm(x)
        x = self.dropout(x, training=training)

        # restore shape
        x = tf.reshape(
            x,
            (B, N, self.lookback, self.embed_dim)
        )

        return x
