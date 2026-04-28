# bpe_tokenizer.py

import os
from tokenizers import ByteLevelBPETokenizer

class BPETokenizer:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))  # project root

        vocab_path = os.path.join(base_dir, "tokenizer", "vocab.json")
        merges_path = os.path.join(base_dir, "tokenizer", "merges.txt")

        print("Loading tokenizer from:", vocab_path)

        self.tokenizer = ByteLevelBPETokenizer(
            vocab_path,
            merges_path
        )

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
    

    
