import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import os
import sys

# Add project root to sys.path (two levels up from current file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from config.config import load_config

# config = load_config()

class PortfolioGenerator(Layer):
    def __init__(self, config):
        super(PortfolioGenerator, self).__init__()
        self.fraction = config.fraction
    
    def call(self, s, mask = None):

        B = tf.shape(s)[0]
        N = tf.shape(s)[1]
        d = tf.shape(s)[2]

        G = tf.maximum(1, tf.cast(tf.math.floor(self.fraction * tf.cast(N, tf.float32)), tf.int32))

        if mask is not None:
            s = tf.where(mask==0, tf.constant(-np.inf, dtype = tf.float32),s)
        
        # first, sort assets
        s_max = tf.reduce_max(s, axis=-1)
        sorted_values, sort_indices = tf.nn.top_k(s_max, k = N, sorted = True)

        # select top G and bottom G asset indices
        top_indices = sort_indices[:, :G] # [B, G]
        bottom_indices = sort_indices[:, -G:] # [B, G]

        # select top and bottom scores (vectors)
        top_scores = tf.gather(s_max, top_indices, batch_dims = 1) # [B, G, d]
        bottom_scores = tf.gather(s_max, bottom_indices, batch_dims = 1) # [B, G, d]

        # calculate top and bottom weights
        top_weights = tf.nn.softmax(top_scores, axis=1)
        bottom_weights = tf.nn.softmax(-bottom_scores, axis = 1)

        portfolio_weights = tf.zeros((B,N), dtype = tf.float32) # [B, N]

        batch_idx = tf.repeat(tf.range(B), G)
        
        # Getting top indexes and values
        top_idx = tf.reshape(top_indices, [-1])
        top_values = tf.reshape(top_weights, [-1])
        scatter_idx_top = tf.stack([batch_idx, top_idx], axis=1)
        portfolio_weights = tf.tensor_scatter_nd_update(portfolio_weights, 
                                                        scatter_idx_top,
                                                        top_values
                                                        )
        
        # Getting bottom indexes and values
        bottom_idx = tf.reshape(bottom_indices, [-1])
        bottom_values = tf.reshape(bottom_weights, [-1])
        scatter_idx_bot = tf.stack([batch_idx, bottom_idx], axis = 1)

        portfolio_weights = tf.tensor_scatter_nd_update(portfolio_weights, 
                                                        scatter_idx_bot,
                                                        -bottom_values
                                                        )
        
        return portfolio_weights, sort_indices
        






        

