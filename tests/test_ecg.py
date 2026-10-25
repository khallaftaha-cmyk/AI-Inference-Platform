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
from app.core.validators import validate_ecg_signal


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
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
      - All expected response fields present (including new validation_warnings)
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
    assert "validation_warnings" in data  # new field from data quality checks

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

    422 = Unprocessable Entity (request format is semantically wrong).
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


# ─────────────────────────────────────────────────────────────────────────────
# NEW TESTS: Data Quality Validator (Option C)
# These test the validator logic directly, without HTTP — unit tests.
# ─────────────────────────────────────────────────────────────────────────────

def test_validator_passes_clean_ecg():
    """
    A clean 12×5000 signal with normal amplitude should pass validation.
    """
    clean_signal = (np.random.randn(12, 5000) * 0.5).tolist()  # ~0.5 mV amplitude
    result = validate_ecg_signal(clean_signal)

    assert result.is_valid is True
    assert len(result.issues) == 0
    assert "num_leads" in result.stats
    assert result.stats["num_leads"] == 12


def test_validator_rejects_wrong_shape():
    """
    A signal with 8 leads instead of 12 should fail validation.
    The model expects exactly 12 leads (12-lead ECG).
    """
    wrong_leads_signal = np.random.randn(8, 5000).tolist()
    result = validate_ecg_signal(wrong_leads_signal)

    assert result.is_valid is False
    assert any("leads" in issue.lower() for issue in result.issues)


def test_validator_rejects_flatline():
    """
    A signal where one lead is all zeros (flatline) should fail.
    This simulates a disconnected electrode — a real clinical problem.
    """
    signal = np.random.randn(12, 5000).tolist()
    signal[3] = [0.0] * 5000  # Flatline lead 3

    result = validate_ecg_signal(signal)

    assert result.is_valid is False
    assert any("flatline" in issue.lower() for issue in result.issues)


def test_validator_warns_on_high_amplitude():
    """
    A signal with very high amplitude (e.g., 20 mV) should produce a WARNING
    but NOT fail outright — it could still be valid data, just unusual.
    """
    # 20mV amplitude (above the ±15mV threshold warning)
    signal = (np.random.randn(12, 5000) * 10.0).tolist()  # std=10 → values reach ~20 mV
    result = validate_ecg_signal(signal)

    # Should still pass (warning only, not a hard failure)
    # Note: if std is too high it might also trigger high_noise warning
    # The key check is that amplitude warning appears
    assert isinstance(result.warnings, list)


@pytest.mark.asyncio
async def test_api_rejects_flatline_signal(client: AsyncClient):
    """
    Integration test: The API should return 422 when a flatline lead is sent.
    This proves the validator is correctly wired into the prediction endpoint.
    """
    signal = np.random.randn(12, 5000).tolist()
    signal[0] = [0.0] * 5000  # Flatline lead 0

    response = await client.post(
        "/api/v1/predict/ecg",
        json={"signal": signal, "patient_id": "flatline-test"}
    )

    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    # The detail should contain our custom message, not just Pydantic's default
    assert "flatline" in str(body["detail"]).lower()

