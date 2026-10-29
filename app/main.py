"""
main.py — The application entry point. Everything starts here.

This file:
  1. Creates the FastAPI app instance.
  2. Uses a "lifespan" function to initialize the ModelRegistry at startup.
  3. Includes the routers (URL handlers) we defined in routers/.
  4. Sets up Prometheus metrics collection.
  5. Configures logging.

Startup sequence:
  uvicorn runs main.py
      → FastAPI app created
      → lifespan() runs (ModelRegistry initializes, models load into memory)
      → Server starts accepting HTTP requests
      → Requests hit routers → routers call model runners → responses returned
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import settings
from app.core.registry import ModelRegistry
from app.routers import ecg, models

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# Configure how log messages look and where they go (stdout/console here).
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,          # Show INFO, WARNING, ERROR, CRITICAL messages
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]  # Print to terminal
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — Startup & Shutdown Logic
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Runs once at startup (before yield) and once at shutdown (after yield).
    """
    # ── STARTUP ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    # Initialize the Dynamic Model Registry
    logger.info("Initializing Model Registry...")
    registry = ModelRegistry(models_dir=settings.MODELS_STORE_DIR)
    app.state.model_registry = registry

    # Pre-warm & load initial default model
    logger.info("Loading default ECG model into registry...")
    try:
        model_v1 = registry.load_model("v1")
        app.state.ecg_model_v1 = model_v1
    except Exception as e:
        logger.error(f"[ERROR] Could not load model v1: {e}")
        app.state.ecg_model_v1 = None

    # For backward-compatibility with canary route
    app.state.ecg_model_v2 = app.state.ecg_model_v1

    logger.info("[OK] Startup complete. Server is ready.")
    logger.info("     API docs available at: http://localhost:8000/docs")

    yield  # ← Server runs here, handling requests until shutdown is triggered

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("Shutting down... cleaning up resources.")
    app.state.ecg_model_v1 = None
    app.state.ecg_model_v2 = None
    app.state.model_registry = None
    logger.info("Goodbye!")


# ─────────────────────────────────────────────────────────────────────────────
# APP CREATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-grade AI Inference Platform serving ECG classification models "
        "with dynamic hot-reloading, canary deployment, and Prometheus observability."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─────────────────────────────────────────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────────────────────────────────────────

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ─────────────────────────────────────────────────────────────────────────────
# INCLUDE ROUTERS
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(ecg.router, prefix=settings.API_PREFIX)
app.include_router(models.router, prefix=settings.API_PREFIX)


# ─────────────────────────────────────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint — confirms the server is running."""
    return {
        "message": f"Welcome to {settings.APP_NAME}!",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/health",
        "models": "/api/models",
    }
