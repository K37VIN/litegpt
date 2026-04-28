import yaml
import tensorflow as tf

from tokenizer.bpe_tokenizer import BPETokenizer
from model.gpt import GPT
from inference.generate import generate


# ------------------------
# Load config
# ------------------------
config = yaml.safe_load(open("configs/config.yaml"))

# ------------------------
# Tokenizer
# ------------------------
tok = BPETokenizer()

# ------------------------
# Load model
# ------------------------
model = GPT(tok.vocab_size(), config["model"])

dummy_input = tf.zeros(
    (1, config["model"]["block_size"]),
    dtype=tf.int32
)
model(dummy_input)

model.load_weights("checkpoints/final_model.weights.h5")


# ------------------------
# Chat loop
# ------------------------
print("\n🤖 MiniGPT Chat Mode (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye 👋")
        break

    # ------------------------
    # STRICT chat format (VERY IMPORTANT)
    # ------------------------
    prompt = f"""User: {user_input}
Assistant:"""

    tokens = tok.encode(prompt)
    tokens = tokens[-config["model"]["block_size"]:]

    prompt_trimmed = tok.decode(tokens)

    output = generate(
        model,
        tok,
        prompt_trimmed,
        max_len=120,
        temperature=0.8,
        block_size=config["model"]["block_size"]
    )

    # extract only assistant response
    reply = output.split("Assistant:")[-1].strip()

    print("\nBot:", reply, "\n")