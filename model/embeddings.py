# embeddings.py

import tensorflow as tf

class Embeddings(tf.keras.layers.Layer):
    def __init__(self,vocab_size,n_embed,block_size):
        super().__init__()
        self.token = tf.keras.layers.Embedding(vocab_size, n_embed)
        self.position = tf.keras.layers.Embedding(block_size, n_embed)
    
    def call(self,idx):
        B,T =idx.shape
        tok = self.token(idx)
        pos = self.position(tf.range(T))
        return tok + pos
    

    

    
