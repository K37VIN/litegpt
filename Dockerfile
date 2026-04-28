# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── HuggingFace Spaces metadata ───────────────────────────────────────────────
# These labels are read by HF Spaces to configure the running app
LABEL org.opencontainers.image.title="LiteGPT Shakespeare API"
LABEL org.opencontainers.image.description="FastAPI + TensorFlow Shakespeare text generator"

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (Docker layer cache) ───────────────────────────────
COPY requirements.txt .

# ── Install Python dependencies ────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy full project ─────────────────────────────────────────────────────────
COPY . .

# ── HuggingFace Spaces runs as non-root user (uid 1000) ───────────────────────
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# ── Expose port 7860 (required by HF Spaces) ──────────────────────────────────
EXPOSE 7860

# ── Start FastAPI with uvicorn on port 7860 ────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]