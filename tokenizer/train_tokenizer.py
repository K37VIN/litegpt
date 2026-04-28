# train_tokenizer.py

from tokenizers import ByteLevelBPETokenizer
import os

def train_tokenizer(files, vocab_size=2000):
    print("Training tokenizer...")

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=files,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"]
    )

    # 🔥 THIS WAS MISSING
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save_model("tokenizer")

    print("Tokenizer saved to tokenizer/ folder")