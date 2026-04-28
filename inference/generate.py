# inference/generate.py

import tensorflow as tf

def generate(model, tokenizer, prompt, max_len=100, temperature=0.8, block_size=128):
    tokens = tokenizer.encode(prompt)

    for _ in range(max_len):
       
        x = tf.constant([tokens[-block_size:]], dtype=tf.int32)

        logits = model(x)

        
        logits = logits[:, -1, :] / temperature

        probs = tf.nn.softmax(logits)

       
        next_token = tf.random.categorical(tf.math.log(probs), 1)
        tokens.append(int(next_token[0][0]))

    return tokenizer.decode(tokens)