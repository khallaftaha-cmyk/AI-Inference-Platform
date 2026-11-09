"""
main.py — The application entry point. Everything starts here.

This file:
  1. Creates the FastAPI app instance.
  2. Uses a "lifespan" function to initialize:
       - ModelRegistry (model versioning and hot-swapping)
       - RetrainingPipeline (automated candidate generation and fine-tuning)
       - DriftDetector (sliding-window quality anomaly detection)
  3. Includes the routers (URL handlers) we defined in routers/.
  4. Sets up Prometheus metrics collection.
  5. Configures logging.

Startup sequence:
  uvicorn runs main.py
      → FastAPI app created
      → lifespan() runs (Registry, Pipeline, and DriftDetector initialize)
      → Server starts accepting HTTP requests
      → Requests hit routers → validators check drift → model runs → response
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import settings
from app.core.drift import DriftDetector
from app.core.registry import ModelRegistry
from app.core.retraining import RetrainingPipeline
from app.routers import drift, ecg, models

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

    # 1. Initialize the Dynamic Model Registry
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

    # 2. Initialize the Automated Retraining Pipeline
    logger.info("Initializing Retraining Pipeline...")
    retraining_pipeline = RetrainingPipeline(
        models_store_dir=settings.MODELS_STORE_DIR,
        model_registry=registry,
    )
    app.state.retraining_pipeline = retraining_pipeline

    # 3. Initialize the Real-Time Drift Detector with Auto-Trigger Callback
    def on_drift_detected_callback(summary: dict):
        failure_rate = summary.get("failure_rate_pct", 0.0)
        threshold = summary.get("threshold_pct", 5.0)
        reason = f"Automated drift alert: {failure_rate:.1f}% failure rate exceeded threshold {threshold:.1f}%"
        logger.warning(f"[AUTO-DRIFT-TRIGGER] {reason}")
        retraining_pipeline.trigger_job(
            trigger_type="auto_drift",
            reason=reason,
            auto_activate=False,
        )

    logger.info("Initializing Drift Detector...")
    drift_detector = DriftDetector(
        window_size=50,
        drift_threshold_pct=5.0,
        min_samples=10,
        on_drift_detected=on_drift_detected_callback,
    )
    app.state.drift_detector = drift_detector

    logger.info("[OK] Startup complete. Server is ready.")
    logger.info("     API docs available at: http://localhost:8000/docs")

    yield  # ← Server runs here, handling requests until shutdown is triggered

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("Shutting down... cleaning up resources.")
    app.state.ecg_model_v1 = None
    app.state.ecg_model_v2 = None
    app.state.model_registry = None
    app.state.retraining_pipeline = None
    app.state.drift_detector = None
    logger.info("Goodbye!")


# ─────────────────────────────────────────────────────────────────────────────
# APP CREATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-grade AI Inference Platform serving ECG classification models "
        "with dynamic hot-reloading, drift detection, automated retraining, and Prometheus observability."
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
app.include_router(drift.router, prefix=settings.API_PREFIX)


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
        "drift": "/api/drift/status",
    }
