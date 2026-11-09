"""
test_drift.py — Tests for Real-Time Data Drift Detection and Automated Retraining.
"""

import time
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from app.core.drift import DriftDetector, DriftSeverity
from app.core.retraining import RetrainingPipeline, RetrainingJobStatus
from app.main import app


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURE
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
async def client():
    """Shared test client with app lifespan context."""
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as c:
            yield c


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS: DriftDetector Engine
# ─────────────────────────────────────────────────────────────────────────────

def test_drift_detector_initial_state():
    """Initial detector should have 0 samples, 0% failure rate, NORMAL status."""
    detector = DriftDetector(window_size=20, drift_threshold_pct=5.0, min_samples=5)
    status = detector.get_status()

    assert status["severity"] == DriftSeverity.NORMAL.value
    assert status["total_samples"] == 0
    assert status["failure_rate_pct"] == 0.0
    assert status["alert_active"] is False


def test_drift_detector_clean_samples():
    """A series of valid records should keep severity NORMAL."""
    detector = DriftDetector(window_size=20, drift_threshold_pct=5.0, min_samples=5)

    for _ in range(10):
        detector.record_validation(is_valid=True)

    status = detector.get_status()
    assert status["severity"] == DriftSeverity.NORMAL.value
    assert status["failure_rate_pct"] == 0.0
    assert status["alert_active"] is False
    assert status["total_samples"] == 10


def test_drift_detector_threshold_breach():
    """When failure rate exceeds threshold (e.g. 20% > 5%), CRITICAL_DRIFT fires."""
    detector = DriftDetector(window_size=20, drift_threshold_pct=5.0, min_samples=5)

    # 8 passes, 2 fails = 20% failure rate (above 5% threshold)
    for _ in range(8):
        detector.record_validation(is_valid=True)
    for _ in range(2):
        detector.record_validation(is_valid=False, issues=["Flatline detected in lead 1"])

    status = detector.get_status()
    assert status["severity"] == DriftSeverity.CRITICAL_DRIFT.value
    assert status["alert_active"] is True
    assert status["failure_rate_pct"] == 20.0
    assert "Flatline detected in lead 1" in status["issue_breakdown"]


def test_drift_detector_callback_execution():
    """DriftDetector should call on_drift_detected when threshold is breached."""
    called = []

    def callback(summary):
        called.append(summary)

    detector = DriftDetector(
        window_size=10,
        drift_threshold_pct=10.0,
        min_samples=5,
        on_drift_detected=callback,
    )

    # Add 4 clean samples
    for _ in range(4):
        detector.record_validation(is_valid=True)

    # 5th sample is a failure -> 20% failure rate with 5 samples -> triggers callback
    detector.record_validation(is_valid=False, issues=["NaN values detected"])

    # Allow thread to execute
    time.sleep(0.1)

    assert len(called) == 1
    assert called[0]["alert_active"] is True
    assert called[0]["failure_rate_pct"] == 20.0


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS: RetrainingPipeline
# ─────────────────────────────────────────────────────────────────────────────

def test_retraining_pipeline_job_lifecycle(tmp_path):
    """Test creating and running a retraining job to completion."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create dummy base onnx model
    base_model = models_dir / "ecg_model_v1.onnx"
    base_model.write_bytes(b"dummy onnx content")

    pipeline = RetrainingPipeline(models_store_dir=str(models_dir))
    job = pipeline.trigger_job(trigger_type="manual", reason="Test run")

    assert job["status"] in [RetrainingJobStatus.PENDING.value, RetrainingJobStatus.RUNNING.value]
    assert job["trigger_type"] == "manual"

    # Wait for the background worker to finish
    time.sleep(1.5)

    completed_job = pipeline.get_job(job["job_id"])
    assert completed_job["status"] == RetrainingJobStatus.COMPLETED.value
    assert completed_job["candidate_model_version"] is not None
    assert "validation_accuracy" in completed_job["metrics"]


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS: Drift & Retraining API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_api_get_drift_status(client: AsyncClient):
    """GET /api/drift/status returns current window metrics."""
    response = await client.get("/api/drift/status")
    assert response.status_code == 200

    data = response.json()
    assert "severity" in data
    assert "failure_rate_pct" in data
    assert "threshold_pct" in data
    assert "alert_active" in data
    assert "issue_breakdown" in data


@pytest.mark.asyncio
async def test_api_trigger_retraining_manual(client: AsyncClient):
    """POST /api/drift/trigger-retraining initiates a job."""
    payload = {"reason": "Cardiologist reported false positives", "auto_activate": False}
    response = await client.post("/api/drift/trigger-retraining", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "job_id" in data
    assert data["trigger_type"] == "manual"
    assert data["reason"] == "Cardiologist reported false positives"


@pytest.mark.asyncio
async def test_api_list_retraining_jobs(client: AsyncClient):
    """GET /api/drift/jobs returns array of all triggered jobs."""
    # First trigger a job
    await client.post("/api/drift/trigger-retraining", json={"reason": "Audit job"})

    response = await client.get("/api/drift/jobs")
    assert response.status_code == 200
    jobs = response.json()
    assert isinstance(jobs, list)
    assert len(jobs) >= 1
    assert "job_id" in jobs[0]


@pytest.mark.asyncio
async def test_api_alertmanager_webhook(client: AsyncClient):
    """POST /api/drift/webhook receives Prometheus Alertmanager alerts and starts a job."""
    webhook_payload = {
        "status": "firing",
        "receiver": "ecg-drift-receiver",
        "alerts": [
          {
            "status": "firing",
            "labels": {
              "alertname": "ECGDataDriftDetected",
              "severity": "warning"
            },
            "annotations": {
              "summary": "High failure rate detected (>5%)"
            }
          }
        ]
    }

    response = await client.post("/api/drift/webhook", json=webhook_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "acknowledged"
    assert "job_id" in data


@pytest.mark.asyncio
async def test_end_to_end_drift_tracking_via_predictions(client: AsyncClient):
    """
    Sending invalid predictions updates the drift detector sliding window
    and increases the failure rate.
    """
    # 1. Send clean prediction
    clean_signal = (np.random.randn(12, 5000) * 0.5).tolist()
    r1 = await client.post("/api/v1/predict/ecg", json={"signal": clean_signal})
    assert r1.status_code == 200

    # 2. Send invalid (flatline) prediction
    flatline_signal = np.random.randn(12, 5000).tolist()
    flatline_signal[0] = [0.0] * 5000
    r2 = await client.post("/api/v1/predict/ecg", json={"signal": flatline_signal})
    assert r2.status_code == 422

    # 3. Check drift status
    status_res = await client.get("/api/drift/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["total_samples"] >= 2
    assert status_data["failed_samples"] >= 1
    assert any("flatline" in k.lower() for k in status_data["issue_breakdown"].keys())
