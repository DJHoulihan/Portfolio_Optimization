import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
import src.model.transformer as tm

class CAAN(Layer):

    def __init__(self, config):
        super(CAAN, self).__init__()

        self.embed_dim = config.srem.embed_dim
        self.attention = tm.MultiHeadSelfAttention(embed_dim = self.embed_dim, num_heads= 1)
        

    def call(self, inputs, training = False):
        """
        inputs:  (Batch (B), Assets (N), embedded dimension (d))
        returns: (B, N, d)
        """
        x = self.attention(inputs, training = training) # same dim, single head

        return x

         
#  s = self.winnerscore(x, training = training)
        


