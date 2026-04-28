# utils.py

import tensorflow as tf

def create_dataset(token_ids, block_size, batch_size):
    def gen():
        for i in range(len(token_ids) - block_size):
            yield (
                token_ids[i:i+block_size],
                token_ids[i+1:i+block_size+1]
            )

    ds = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.int32,tf.int32)
    )

    return ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

