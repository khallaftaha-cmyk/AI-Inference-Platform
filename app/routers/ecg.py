"""
ecg.py — API routes (endpoints) for ECG prediction.

Key concepts:
  - Router: A FastAPI "mini-app" that groups related endpoints together.
    We then "include" this router in main.py.
    This keeps the code organized — instead of one giant file.

  - Pydantic Models (schemas): Define the shape of request and response JSON.
    FastAPI uses these to:
      1. Validate incoming requests (reject bad data automatically)
      2. Generate OpenAPI documentation (the /docs page)
      3. Serialize response data to JSON

  - Dynamic Model Resolution:
    Predictions look up the currently active model from `request.app.state.model_registry`.
    This enables zero-downtime hot swapping: changing the active model via the registry
    immediately shifts production predictions without restarting the application.
"""

import logging
import random
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from typing import Optional

from app.core.validators import validate_ecg_signal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# REQUEST & RESPONSE SCHEMAS (Pydantic Models)
# ─────────────────────────────────────────────────────────────────────────────

class ECGPredictRequest(BaseModel):
    """
    Defines what the client must send in the request body (JSON).

    Example JSON a client would send:
    {
        "signal": [[0.1, 0.2, -0.1, ...], [0.3, 0.1, 0.0, ...], ...],
        "patient_id": "patient-001"
    }
    """
    signal: list[list[float]] = Field(
        ...,
        description="2D ECG signal: list of leads, each lead is a list of samples.",
    )

    patient_id: Optional[str] = Field(
        default=None,
        description="Optional patient identifier for logging.",
    )

    @field_validator("signal")
    @classmethod
    def validate_signal(cls, v):
        if len(v) == 0:
            raise ValueError("Signal must have at least one lead.")
        for i, lead in enumerate(v):
            if len(lead) < 10:
                raise ValueError(f"Lead {i} has fewer than 10 samples — that's too short.")
        return v


class PredictionResult(BaseModel):
    """
    Defines what the API sends BACK to the client.
    """
    predicted_class: int = Field(description="Index of the predicted class.")
    confidence: float = Field(description="Confidence score (0.0 to 1.0).")
    probabilities: list[float] = Field(description="Probability for each class.")
    model_version: str = Field(description="Which model version made this prediction.")
    patient_id: Optional[str] = Field(default=None)
    validation_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal data quality warnings (e.g., noisy leads, amplitude drift)."
    )


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""
    status: str
    model_v1_loaded: bool
    model_v2_loaded: bool
    active_model_version: Optional[str] = None
    loaded_models_count: int = 0
    app_version: str


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter(tags=["ECG Prediction"])


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: GET /health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Health check endpoint: reports system liveness and model readiness.
    """
    state = request.app.state
    from app.core.config import settings

    registry = getattr(state, "model_registry", None)
    active_version = registry.active_version if registry else None
    loaded_count = len(registry.loaded_versions) if registry else 0

    return HealthResponse(
        status="ok",
        model_v1_loaded=getattr(state, "ecg_model_v1", None) is not None,
        model_v2_loaded=getattr(state, "ecg_model_v2", None) is not None,
        active_model_version=active_version,
        loaded_models_count=loaded_count,
        app_version=settings.APP_VERSION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: POST /v1/predict/ecg
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/v1/predict/ecg", response_model=PredictionResult)
async def predict_ecg_v1(body: ECGPredictRequest, request: Request):
    """
    Predict ECG class using the active production model.

    Dynamic Model Resolution:
      Pulls the active model from ModelRegistry. If a model was hot-swapped
      live via /api/models/{version}/activate, this endpoint automatically uses it!
    """
    registry = getattr(request.app.state, "model_registry", None)
    if registry:
        model = registry.get_active_model()
    else:
        model = getattr(request.app.state, "ecg_model_v1", None)

    if model is None:
        raise HTTPException(status_code=503, detail="Active ECG model is not loaded.")

    logger.info(f"[{model.model_version}] Predicting for patient: {body.patient_id}")

    result = _run_inference(model, body)
    return PredictionResult(**result, patient_id=body.patient_id)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: POST /v2/predict/ecg  (Canary endpoint)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/v2/predict/ecg", response_model=PredictionResult)
async def predict_ecg_v2(body: ECGPredictRequest, request: Request):
    """
    Canary endpoint: routes a percentage of traffic to candidate v2 vs stable v1.
    """
    from app.core.config import settings

    use_v2 = random.random() < settings.CANARY_TRAFFIC_RATIO

    if use_v2:
        model = getattr(request.app.state, "ecg_model_v2", None)
        if model is None:
            raise HTTPException(status_code=503, detail="ECG model v2 is not loaded.")
        logger.info(f"[canary→v2] Predicting for patient: {body.patient_id}")
    else:
        registry = getattr(request.app.state, "model_registry", None)
        model = registry.get_active_model() if registry else getattr(request.app.state, "ecg_model_v1", None)
        if model is None:
            raise HTTPException(status_code=503, detail="ECG model v1 is not loaded.")
        logger.info(f"[canary→v1] Predicting for patient: {body.patient_id}")

    result = _run_inference(model, body)
    return PredictionResult(**result, patient_id=body.patient_id)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION (private)
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference(model, body: ECGPredictRequest) -> dict:
    """
    Shared inference logic used by both v1 and v2 endpoints.
    """
    # Step 1: Data quality validation gate
    validation = validate_ecg_signal(
        signal=body.signal,
        patient_id=body.patient_id,
    )

    if not validation.is_valid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "ECG signal failed data quality validation.",
                "issues": validation.issues,
                "hint": "Check signal shape (12 leads × 5000 samples), units (mV), and electrode connections."
            }
        )

    # Step 2: Convert to numpy array with batch dimension (1, 12, 5000)
    signal_array = np.array(body.signal, dtype=np.float32)[np.newaxis, ...]

    # Step 3: Run model inference
    try:
        result = model.predict(signal_array)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Step 4: Attach validation warnings to result
    result["validation_warnings"] = validation.warnings
    return result
