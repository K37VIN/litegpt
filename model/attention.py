# attention.py

import tensorflow as tf

def causal_mask(T):
    return tf.linalg.band_part(tf.ones((T,T)),-1,0)

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self,n_heads,head_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size

        self.key = tf.keras.layers.Dense(n_heads * head_size)
        self.query = tf.keras.layers.Dense(n_heads * head_size)
        self.value = tf.keras.layers.Dense(n_heads * head_size)

        self.proj = tf.keras.layers.Dense(n_heads * head_size)

    def call(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = tf.reshape(k, (B, T, self.n_heads, self.head_size))
        q = tf.reshape(q, (B, T, self.n_heads, self.head_size))
        v = tf.reshape(v, (B, T, self.n_heads, self.head_size))
        
        k = tf.transpose(k, [0,2,1,3])
        q = tf.transpose(q, [0,2,1,3])
        v = tf.transpose(v, [0,2,1,3])

        wei = tf.matmul(q,k ,transpose_b=True)
        wei = wei / tf.math.sqrt(tf.cast(self.head_size, tf.float32))

        mask = causal_mask(T)
        wei = tf.where(mask==0, -1e9, wei)

        wei = tf.nn.softmax(wei, axis=-1)

        out = tf.matmul(wei, v)

        out = tf.transpose(out,[0,2,1,3])
        out = tf.reshape(out, (B, T, self.n_heads*self.head_size))

        return self.proj(out)
        
        

