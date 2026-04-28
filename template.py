import os

STRUCTURE = {
    "configs": ["config.yaml"],
    "data/raw": [],
    "data/processed": [],
    "tokenizer": ["bpe_tokenizer.py", "train_tokenizer.py"],
    "model": ["attention.py", "block.py", "gpt.py", "embeddings.py"],
    "training": ["train.py", "trainer.py", "utils.py"],
    "inference": ["generate.py"],
    "vector_db": ["index.py", "retriever.py"],
    "": ["main.py", "requirements.txt"]
}


DEFAULT_CONTENT = {
    "config.yaml": """model:
  n_layers: 4
  n_heads: 8
  n_embed: 256
  block_size: 128

training:
  batch_size: 32
  lr: 3e-4
  steps: 5000

data:
  dataset_path: "data/raw/input.txt"

tokenizer:
  vocab_size: 5000
""",

    "requirements.txt": """tensorflow
numpy
pyyaml
tokenizers
""",

    "main.py": """def main():
    print("Mini GPT Project Initialized")

if __name__ == "__main__":
    main()
"""
}

def create_structure(base_path="."):
    for folder, files in STRUCTURE.items():
        folder_path = os.path.join(base_path, folder)

        # Create folder
        if folder and not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"📁 Created folder: {folder_path}")

        # Create files
        for file in files:
            file_path = os.path.join(folder_path, file)

            if not os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    content = DEFAULT_CONTENT.get(file, f"# {file}\n")
                    f.write(content)
                print(f"📄 Created file: {file_path}")
            else:
                print(f"⚠️ Skipped (exists): {file_path}")

if __name__ == "__main__":
    create_structure()

