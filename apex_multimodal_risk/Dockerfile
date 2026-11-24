FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user setup
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy lean serving dependency specification and install
COPY requirements-serving.txt /app/apex_multimodal_risk/requirements-serving.txt
RUN pip install --no-cache-dir -r /app/apex_multimodal_risk/requirements-serving.txt

# Pre-fetch pure Rust tokenizer.json so runtime container boots 100% air-gapped
RUN python -c "from tokenizers import Tokenizer; import os; os.makedirs('/app/apex_multimodal_risk/artifacts/onnx', exist_ok=True); tok = Tokenizer.from_pretrained('BAAI/bge-small-en-v1.5'); tok.save('/app/apex_multimodal_risk/artifacts/onnx/tokenizer.json')"

# Copy application source tree and configurations
COPY . /app/apex_multimodal_risk/

# Set ownership to appuser
RUN chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "apex_multimodal_risk.src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
