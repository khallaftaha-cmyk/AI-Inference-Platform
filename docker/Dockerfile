# ── Stage 1: Base Image ────────────────────────────────────────────────────────
# python:3.11-slim provides a lightweight Debian Linux container (~150MB) with Python 3.11.
# Pre-compiled wheels for onnxruntime and numpy exist for Python 3.11 Linux,
# ensuring fast and reliable installation without needing C++ compilers.
FROM python:3.11-slim

# ── Environment variables for Python inside containers ────────────────────────
# PYTHONUNBUFFERED=1: Flushes stdout and stderr immediately (no buffering).
#                     Ensures logs appear in `docker logs` in real-time.
# PYTHONDONTWRITEBYTECODE=1: Prevents Python from writing .pyc files to disk.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Install curl for container health checks ──────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory inside container ────────────────────────────────────
WORKDIR /app

# ── Install dependencies first (Docker Layer Caching) ─────────────────────────
# By copying requirements.txt alone first, Docker caches this layer.
# Re-building the image after changing code will NOT re-run pip install!
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code and model store ─────────────────────────────────────
COPY app/ ./app/
COPY models_store/ ./models_store/

# ── Document exposed port ─────────────────────────────────────────────────────
EXPOSE 8000

# ── Container Healthcheck ─────────────────────────────────────────────────────
# Periodically queries /api/health to confirm the service is responsive and models are loaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# ── Container entrypoint command ──────────────────────────────────────────────
# Listens on 0.0.0.0 (required for Docker containers to be reachable from outside).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]