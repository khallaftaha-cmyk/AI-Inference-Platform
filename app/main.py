"""
main.py — The application entry point. Everything starts here.

This file:
  1. Creates the FastAPI app instance.
  2. Uses a "lifespan" function to load models at startup and clean up at shutdown.
  3. Includes the routers (URL handlers) we defined in routers/.
  4. Sets up Prometheus metrics collection.
  5. Configures logging.

Startup sequence:
  uvicorn runs main.py
      → FastAPI app created
      → lifespan() runs (models load into memory)
      → Server starts accepting HTTP requests
      → Requests hit routers → routers call model runners → responses returned
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import settings
from app.models.ecg_runner import ECGModelRunner
from app.routers import ecg

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# Configure how log messages look and where they go (stdout/console here).
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,          # Show INFO, WARNING, ERROR, CRITICAL messages
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    # Example output: "2026-10-11 20:00:01 | INFO     | app.models.ecg_runner | [OK] Model loaded"
    handlers=[logging.StreamHandler(sys.stdout)]  # Print to terminal
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — Startup & Shutdown Logic
#
# Why lifespan instead of @app.on_event("startup")?
#   FastAPI recommends the lifespan approach (Python 3.10+).
#   It uses a context manager so startup and shutdown are in one function,
#   making it cleaner and easier to reason about.
#
# The `yield` keyword splits the function in two:
#   - Everything BEFORE yield = startup (runs when server starts)
#   - Everything AFTER yield  = shutdown (runs when server stops)
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

    # Load ECG model v1 and store it in app.state so all routes can access it.
    # app.state is a simple namespace — you can attach any Python object to it.
    # Routes access it via `request.app.state.ecg_model_v1`
    logger.info("Loading ECG model v1...")
    try:
        app.state.ecg_model_v1 = ECGModelRunner(
            model_path=settings.ECG_MODEL_V1_PATH,
            model_version="v1"
        )
    except FileNotFoundError as e:
        logger.error(f"[ERROR] Could not load model v1: {e}")
        app.state.ecg_model_v1 = None

    # Load ECG model v2 (canary model — currently same file as v1).
    logger.info("Loading ECG model v2 (canary)...")
    try:
        app.state.ecg_model_v2 = ECGModelRunner(
            model_path=settings.ECG_MODEL_V2_PATH,
            model_version="v2"
        )
    except FileNotFoundError as e:
        logger.error(f"[ERROR] Could not load model v2: {e}")
        app.state.ecg_model_v2 = None

    logger.info("[OK] Startup complete. Server is ready.")
    logger.info("     API docs available at: http://localhost:8000/docs")

    yield  # ← Server runs here, handling requests until shutdown is triggered

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("Shutting down... cleaning up resources.")
    # In a real app, you'd close database connections, flush logs, etc. here.
    app.state.ecg_model_v1 = None
    app.state.ecg_model_v2 = None
    logger.info("Goodbye! Goodbye!")


# ─────────────────────────────────────────────────────────────────────────────
# APP CREATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-grade AI Inference Platform serving ECG classification models "
        "with versioned endpoints, canary deployment, and Prometheus observability."
    ),
    # lifespan= wires up our startup/shutdown function defined above
    lifespan=lifespan,
    # Where the auto-generated docs UI lives.
    # Visit http://localhost:8000/docs to see a clickable UI for all endpoints!
    docs_url="/docs",
    redoc_url="/redoc",  # Alternative docs UI
)


# ─────────────────────────────────────────────────────────────────────────────
# PROMETHEUS METRICS
#
# This one line adds automatic tracking of:
#   - Total request count (by endpoint, method, status code)
#   - Request latency (in seconds, as a histogram)
#
# These metrics are exposed at GET /metrics in a format Prometheus can scrape.
# We'll hook up Prometheus + Grafana in Phase 2 to visualize them.
# ─────────────────────────────────────────────────────────────────────────────

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ─────────────────────────────────────────────────────────────────────────────
# INCLUDE ROUTERS
#
# "Including" a router = telling FastAPI "all endpoints defined in ecg.router
# are now part of this app, prefixed with /api".
#
# So:
#   ecg.router has  GET  /health
#                   POST /v1/predict/ecg
#                   POST /v2/predict/ecg
#
# After include_router, they become:
#   GET  /api/health
#   POST /api/v1/predict/ecg
#   POST /api/v2/predict/ecg
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(ecg.router, prefix=settings.API_PREFIX)


# ─────────────────────────────────────────────────────────────────────────────
# ROOT ENDPOINT — a simple welcome message
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint — confirms the server is running."""
    return {
        "message": f"Welcome to {settings.APP_NAME}!",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/health",
    }
