import yaml
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from tokenizer.bpe_tokenizer import BPETokenizer
from model.gpt import GPT
from inference.generate import generate

# -------------------------
# Load config
# -------------------------
config = yaml.safe_load(open("configs/config.yaml"))

# -------------------------
# Load tokenizer
# -------------------------
tok = BPETokenizer()

# -------------------------
# Load model
# -------------------------
model = GPT(tok.vocab_size(), config["model"])

# Build model (required in TF)
dummy = tf.zeros(
    (1, config["model"]["block_size"]),
    dtype=tf.int32
)
model(dummy)

# Load trained weights
model.load_weights("checkpoints/final_model.weights.h5")

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="LiteGPT Shakespeare API")

# -------------------------
# Enable CORS (frontend connect)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request schema
# -------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_len: int = 200       # matches frontend default
    temperature: float = 0.8  # matches frontend default


# -------------------------
# Health check
# -------------------------
@app.get("/")
def home():
    return {"status": "LiteGPT Shakespeare API running"}


# -------------------------
# Generation endpoint
# -------------------------
@app.post("/generate")
def generate_text(req: GenerateRequest):
    # Structured Shakespeare-style prompt
    prompt = f"Continue in Shakespearean style:\n{req.prompt}\n"

    output = generate(
        model=model,
        tokenizer=tok,
        prompt=prompt,
        max_len=req.max_len,
        temperature=req.temperature,
        block_size=config["model"]["block_size"]
    )

    # Remove the injected prompt prefix, return only generated text
    result = output.replace(prompt, "").strip()

    return {
        "input": req.prompt,
        "output": result
    }