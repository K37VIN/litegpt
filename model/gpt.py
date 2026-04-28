# gpt.py

import tensorflow as tf
from .block import Block
from .embeddings import Embeddings

class GPT(tf.keras.Model):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.embed = Embeddings(
            vocab_size,
            config["n_embed"],
            config["block_size"]
        )

        self.blocks = [
            Block(config["n_embed"], config["n_heads"])
            for _ in range(config["n_layers"])
        ]

        self.ln_f = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(vocab_size)

    def call(self, idx):
        x = self.embed(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x)