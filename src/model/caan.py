import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
import transformer as tm
import os
import sys

class WinnerScore(Layer):
    def __init__(self, embed_dim):
        super(WinnerScore, self).__init__()

        self.w_s = self.add_weight(
            shape = (embed_dim, 1),
            initializer = 'glorot_uniform',
            trainable = True,
            name = 'w_s'
        )

        self.e_s = self.add_weight(
            shape = (1,),
            initializer = 'glorot_uniform',
            trainable = True,
            name = 'e_s'
        )
    
    def call(self, a):
        
        s = tf.matmul(a, self.w_s) + self.e_s
        s = tf.tanh(s)
        s = tf.squeeze(s, axis = -1)
    
        return s

class CAAN(Layer):

    def __init__(self, config):
        super(CAAN, self).__init__()

        self.embed_dim = config.embed_dim
        self.attention = tm.MultiHeadSelfAttention(embed_dim = self.embed_dim, num_heads= 1)
        self.winnerscore = WinnerScore(self.embed_dim)

    def call(self, inputs, training = False):
        """
        inputs:  (Batch (B), Assets (N), embedded dimension (d))
        returns: (B, N, d)
        """
        x = self.attention(inputs, training = training) # same dim, single head

        s = self.winnerscore(x, training = training)

        return s

         

        


