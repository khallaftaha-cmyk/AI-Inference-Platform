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

  - Dependency Injection: The `request: Request` parameter FastAPI passes
    automatically. We use this to access the app state (loaded models).
"""

import logging
import random
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# REQUEST & RESPONSE SCHEMAS (Pydantic Models)
# These define exactly what JSON the API accepts and returns.
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

    # signal: a 2D list — num_leads rows × signal_length columns
    # e.g., 12 rows × 1000 columns for a 12-lead, 1000-sample ECG
    signal: list[list[float]] = Field(
        ...,  # "..." means this field is REQUIRED (no default)
        description="2D ECG signal: list of leads, each lead is a list of samples.",
    )

    # Optional patient ID — for logging/tracking purposes.
    patient_id: Optional[str] = Field(
        default=None,
        description="Optional patient identifier for logging.",
    )

    # Field validator: runs after the basic type check.
    # Here we validate that the signal has at least 1 lead and 10 samples.
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

    Example response:
    {
        "predicted_class": 0,
        "confidence": 0.94,
        "probabilities": [0.94, 0.03, 0.02, 0.01],
        "model_version": "v1",
        "patient_id": "patient-001"
    }
    """
    predicted_class: int = Field(description="Index of the predicted class.")
    confidence: float = Field(description="Confidence score (0.0 to 1.0).")
    probabilities: list[float] = Field(description="Probability for each class.")
    model_version: str = Field(description="Which model version made this prediction.")
    patient_id: Optional[str] = Field(default=None)


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""
    status: str
    model_v1_loaded: bool
    model_v2_loaded: bool
    app_version: str


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

# APIRouter groups these endpoints together.
# "tags" just adds a label in the auto-generated /docs page.
router = APIRouter(tags=["ECG Prediction"])


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: GET /health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Health check endpoint.

    Why do we need this?
      - Kubernetes, Docker, and load balancers call /health to check if
        the app is running and models are loaded before sending real traffic.
      - If a pod crashes or a model fails to load, /health returns a non-200
        status and the orchestrator can restart the pod automatically.

    `request.app.state` is where we store shared objects (like loaded models).
    We set these up in main.py's lifespan function.
    """
    state = request.app.state
    from app.core.config import settings

    return HealthResponse(
        status="ok",
        model_v1_loaded=getattr(state, "ecg_model_v1", None) is not None,
        model_v2_loaded=getattr(state, "ecg_model_v2", None) is not None,
        app_version=settings.APP_VERSION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: POST /v1/predict/ecg
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/v1/predict/ecg", response_model=PredictionResult)
async def predict_ecg_v1(body: ECGPredictRequest, request: Request):
    """
    Predict ECG class using model v1 (stable / production model).

    How a request flows through this function:
      1. Client sends POST /v1/predict/ecg with JSON body.
      2. FastAPI deserializes JSON → ECGPredictRequest (Pydantic validates it).
      3. We convert the 2D list to a numpy array.
      4. We call the model runner's .predict() method.
      5. We return a PredictionResult (Pydantic serializes it back to JSON).
    """
    model = getattr(request.app.state, "ecg_model_v1", None)
    if model is None:
        # 503 = "Service Unavailable" — the server is up but not ready.
        raise HTTPException(status_code=503, detail="ECG model v1 is not loaded.")

    logger.info(f"[v1] Predicting for patient: {body.patient_id}")

    result = _run_inference(model, body)
    return PredictionResult(**result, patient_id=body.patient_id)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: POST /v2/predict/ecg  (Canary endpoint)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/v2/predict/ecg", response_model=PredictionResult)
async def predict_ecg_v2(body: ECGPredictRequest, request: Request):
    """
    Canary endpoint: routes traffic between v1 and v2 based on the ratio.

    What is a Canary Deployment?
      Imagine you've trained a new model (v2). You don't want to switch all
      1,000,000 daily predictions to v2 at once — what if it's worse?

      So instead, you send:
        - 90% of requests → v1 (old, trusted model)
        - 10% of requests → v2 (new, candidate model)

      You watch the metrics. If v2 looks good → increase to 50%, then 100%.
      If v2 looks bad → set ratio back to 0%, no users were harmed.

      This is "canary" because like a canary in a coal mine, you send a small
      batch first to detect problems before they affect everyone.
    """
    from app.core.config import settings

    # random.random() returns a float between 0.0 and 1.0
    # If CANARY_TRAFFIC_RATIO = 0.1, then 10% of the time we use v2.
    use_v2 = random.random() < settings.CANARY_TRAFFIC_RATIO

    if use_v2:
        model = getattr(request.app.state, "ecg_model_v2", None)
        if model is None:
            raise HTTPException(status_code=503, detail="ECG model v2 is not loaded.")
        logger.info(f"[canary→v2] Predicting for patient: {body.patient_id}")
    else:
        model = getattr(request.app.state, "ecg_model_v1", None)
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

    Why extract this into a helper?
      DRY principle — "Don't Repeat Yourself."
      Both endpoints do the same numpy conversion + predict call.
      Having it once means fixing a bug here fixes it for both.
    """
    # Convert the list of lists → 3D numpy array: (batch, leads, samples)
    # np.array(body.signal)  → shape (num_leads, signal_length)
    # [np.newaxis, ...]      → shape (1, num_leads, signal_length) — adds batch dimension
    signal_array = np.array(body.signal, dtype=np.float32)[np.newaxis, ...]
    # Now shape is (1, 12, 1000) — batch_size=1, 12 leads, 1000 samples

    try:
        return model.predict(signal_array)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        # 500 = "Internal Server Error"
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
