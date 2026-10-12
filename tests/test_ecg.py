"""
test_ecg.py — Automated tests for the ECG prediction API.

Key concept — how tests work with FastAPI:
  We use httpx.AsyncClient with `app=app` to talk to our FastAPI app
  directly in memory (no real server needed).

  IMPORTANT: Passing `app=app` triggers the app's lifespan context
  (startup/shutdown). This means models get loaded before tests run.
  Without it, models never load and all model-dependent tests fail with 503.

Tools used:
  - pytest: Test runner
  - httpx.AsyncClient: Sends HTTP requests to the app in memory
  - pytest-asyncio: Allows async test functions
"""

import numpy as np
import pytest
from httpx import AsyncClient

from app.main import app


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def make_fake_ecg(num_leads: int = 12, signal_length: int = 5000) -> list[list[float]]:
    """
    Creates a fake ECG signal for testing (12 leads × 5000 samples).
    The real model expects shape (batch=1, leads=12, samples=5000).
    np.random.randn generates random normal numbers — good enough for testing.
    """
    return np.random.randn(num_leads, signal_length).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED CLIENT FIXTURE
# A pytest fixture is reusable setup code — injected into test functions
# as a parameter (pytest finds them by matching parameter names).
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
async def client():
    """
    Async HTTP client fixture that properly triggers the FastAPI lifespan.

    In FastAPI 0.143+, we use `app.router.lifespan_context(app)` as an
    async context manager to run the startup/shutdown lifecycle.

    How it works:
      1. `app.router.lifespan_context(app)` runs our lifespan() function
         up to the `yield` — this loads the models into app.state.
      2. The httpx client then sends requests to the running app.
      3. After the test, lifespan continues past `yield` → shutdown cleanup.
    """
    from httpx import AsyncClient, ASGITransport

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as c:
            yield c


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — each function uses the `client` fixture as a parameter
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """
    Test the root / endpoint returns 200 and expected JSON fields.
    Status codes: 200=OK, 404=Not Found, 500=Internal Server Error
    """
    response = await client.get("/")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "message" in data
    assert "docs" in data


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """
    Test /api/health endpoint.
    Expects status='ok' and model_v1_loaded=True (real .onnx file is present).
    """
    response = await client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_v1_loaded"] is True, (
        "Model v1 should be loaded! Check that models_store/ecg_model_v1.onnx exists."
    )


@pytest.mark.asyncio
async def test_predict_ecg_v1_valid_input(client: AsyncClient):
    """
    Test POST /api/v1/predict/ecg with a valid fake ECG signal.
    Checks:
      - Status 200
      - All expected response fields present
      - Confidence is between 0 and 1
      - Probabilities sum to ~1.0
      - Model version is 'v1'
    """
    fake_signal = make_fake_ecg()  # 12 leads × 5000 samples
    request_body = {"signal": fake_signal, "patient_id": "test-patient-001"}

    response = await client.post("/api/v1/predict/ecg", json=request_body)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    # Check all expected fields exist
    assert "predicted_class" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "model_version" in data

    # Confidence should be a valid probability (0.0 to 1.0)
    assert 0.0 <= data["confidence"] <= 1.0, f"Confidence {data['confidence']} out of range"

    # Probabilities should sum to ~1.0 (allow tiny floating point rounding error)
    total = sum(data["probabilities"])
    assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"

    # Model version should be labeled 'v1'
    assert data["model_version"] == "v1"


@pytest.mark.asyncio
async def test_predict_ecg_v1_invalid_input(client: AsyncClient):
    """
    Test that the API correctly rejects invalid input.
    We send an empty signal list — Pydantic should auto-reject it with 422.

    422 = Unprocessable Entity (request format is structurally wrong).
    FastAPI + Pydantic handle this automatically — our code never even runs.
    """
    bad_request = {"signal": []}  # Empty signal — validator should reject this

    response = await client.post("/api/v1/predict/ecg", json=bad_request)

    assert response.status_code == 422, f"Expected 422, got {response.status_code}"


@pytest.mark.asyncio
async def test_metrics_endpoint_exists(client: AsyncClient):
    """
    Test that the Prometheus /metrics endpoint responds.
    In Phase 2, Prometheus will call this every 15 seconds to collect metrics.
    """
    response = await client.get("/metrics")

    assert response.status_code == 200
    # Prometheus metrics are in plain-text format
    assert "text/plain" in response.headers.get("content-type", "")
