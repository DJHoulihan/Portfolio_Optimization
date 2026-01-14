import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
import logging

class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads = 8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim) # Q dense layer (what I'm looking for)
        self.key_dense = Dense(embed_dim) # K dense layer (what each position offers)
        self.value_dense = Dense(embed_dim) # V dense layer (what I will pass if the match is strong)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value, mask=None):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # Apply causal mask, it prevents look-ahead bias by restricting the model to looking only at 
        # past and current attention scores.
        if mask is not None:
            scaled_score += (mask * -1e9)

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm = [0,2,1,3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        query = self.query_dense(inputs)
        key   = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.split_heads(query, batch_size)   # (B, H, T, D)
        key   = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Causal mask - Not used here
        # mask shape: (1, 1, T, T)
        # upper triangular (future positions) = 1 -> block them
        # causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len),dtype=tf.float32), -1, 0)
        # causal_mask = 1 - causal_mask
        # causal_mask = tf.reshape(causal_mask, (1, 1, seq_len, seq_len))

        attention_output, _ = self.attention(query, key, value, mask=None)

        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, seq_len, self.embed_dim))
        return self.combine_heads(concat_attention)

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate = 0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim,num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation= 'relu'),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training = False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(self.embed_dim)[tf.newaxis, :], tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(self.embed_dim, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return inputs + pos_encoding

class TransformerEncoder(Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        # logging.info(f"Initializing Transformer Encoder with d_model={embed_dim}, nhead={num_heads}, num_layers={num_layers}")

        self.enc_layers = [TransformerBlock(embed_dim, num_heads, ff_dim, rate)
                           for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, inputs, training=False):
        x = inputs
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training)
        return x