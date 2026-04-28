import tensorflow as tf
import yaml

from tokenizer.bpe_tokenizer import BPETokenizer
from model.gpt import GPT
from inference.generate import generate


config = yaml.safe_load(open("configs/config.yaml"))


tok = BPETokenizer()


model = GPT(tok.vocab_size(), config["model"])


# Build model
dummy_input = tf.zeros((1, config["model"]["block_size"]), dtype=tf.int32)
model(dummy_input)

model.load_weights("checkpoints/final_model.weights.h5")

prompts = [
    "To be or not to be",
    "My lord,",
    "Love is",
    "King:"
]

for p in prompts:
    print("\n" + "="*50)
    print("Prompt:", p)
    print("-"*50)
    print(generate(model, tok, p, max_len=100))