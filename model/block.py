# block.py

import tensorflow as tf
from .attention import MultiHeadSelfAttention

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, n_embed):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(4*n_embed,activation=tf.nn.gelu),
            tf.keras.layers.Dense(n_embed)
        ])

    def call(self, x):
        return self.net(x)
    

class Block(tf.keras.layers.Layer):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.sa=MultiHeadSelfAttention(n_heads, n_embed // n_heads)
        self.ffwd=FeedForward(n_embed)

        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

    

