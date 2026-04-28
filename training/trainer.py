# trainer.py

import tensorflow as tf

class Trainer:
    def __init__(self,model,lr):
        self.model = model
        self.opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y, logits, from_logits=True
                )
            )

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss