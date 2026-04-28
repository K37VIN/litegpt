import yaml
import numpy as np
import os
import sys
import tensorflow as tf

from tokenizer.bpe_tokenizer import BPETokenizer
from model.gpt import GPT
from training.trainer import Trainer
from training.utils import create_dataset

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load config
config = yaml.safe_load(open("configs/config.yaml"))

# Tokenizer
tok = BPETokenizer()

# Load data
text = open(config["data"]["dataset_path"]).read()
tokens = np.array(tok.encode(text), dtype=np.int32)

# Dataset
ds = create_dataset(
    tokens,
    config["model"]["block_size"],
    config["training"]["batch_size"]
)

# Model
model = GPT(tok.vocab_size(), config["model"])
trainer = Trainer(model, float(config["training"]["lr"]))

os.makedirs("checkpoints", exist_ok=True)


for step, (x, y) in enumerate(ds.take(config["training"]["steps"])):
    loss = trainer.train_step(x, y)

    if step % 100 == 0:
        print(f"Step {step}, Loss {loss.numpy()}")

    
    if step % 500 == 0 and step > 0:
        model.save_weights("checkpoints/model.weights.h5")
        print(f" Model saved at step {step}")


model.save_weights("checkpoints/final_model.weights.h5")
print(" Final model saved")