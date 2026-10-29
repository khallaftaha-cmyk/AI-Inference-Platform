"""
test_models.py — Tests for the Model Registry and Lifecycle Management endpoints.
"""

import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
async def client():
    """Shared test client with app lifespan context."""
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as c:
            yield c


@pytest.mark.asyncio
async def test_list_models(client: AsyncClient):
    """Test GET /api/models returns model catalog and in-memory status."""
    response = await client.get("/api/models")
    assert response.status_code == 200

    data = response.json()
    assert "total_models" in data
    assert "active_version" in data
    assert "loaded_versions" in data
    assert "models" in data

    assert data["total_models"] >= 1
    assert data["active_version"] is not None
    assert len(data["loaded_versions"]) >= 1

    first_model = data["models"][0]
    assert "version" in first_model
    assert "file_size_mb" in first_model
    assert "is_loaded" in first_model
    assert "is_active" in first_model


@pytest.mark.asyncio
async def test_load_nonexistent_model(client: AsyncClient):
    """Attempting to load a non-existent model should return 404."""
    response = await client.post("/api/models/nonexistent_model_xyz/load")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_activate_model(client: AsyncClient):
    """Test activating a model updates the active_version."""
    response = await client.post("/api/models/v1/activate")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "v1" in data["active_version"]


@pytest.mark.asyncio
async def test_cannot_unload_active_model(client: AsyncClient):
    """Attempting to unload the currently active model must fail with 400."""
    # First ensure v1 is active
    await client.post("/api/models/v1/activate")

    # Try to unload the active model
    response = await client.post("/api/models/v1/unload")
    assert response.status_code == 400
    assert "cannot unload active model" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_health_reports_registry_status(client: AsyncClient):
    """Health endpoint should report active model version and count of loaded models."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "active_model_version" in data
    assert "loaded_models_count" in data
    assert data["loaded_models_count"] >= 1


@pytest.mark.asyncio
async def test_end_to_end_hot_swap_workflow(client: AsyncClient):
    """
    Test zero-downtime model promotion:
    1. Active model is v1. Predict returns model_version 'v1'.
    2. Pre-warm / load v2 into RAM without disrupting traffic.
    3. Activate v2.
    4. Predict now returns model_version 'v2' instantly!
    5. Roll back to v1.
    """
    fake_signal = (np.random.randn(12, 5000) * 0.5).tolist()

    # 1. Ensure v1 is active
    await client.post("/api/models/v1/activate")
    r1 = await client.post("/api/v1/predict/ecg", json={"signal": fake_signal})
    assert r1.status_code == 200
    assert r1.json()["model_version"] == "v1"

    # 2. Pre-warm v2 into memory
    load_res = await client.post("/api/models/v2/load")
    assert load_res.status_code == 200
    assert "v2" in load_res.json()["loaded_versions"]

    # 3. Hot swap / promote v2 to production
    act_res = await client.post("/api/models/v2/activate")
    assert act_res.status_code == 200
    assert act_res.json()["active_version"] == "v2"

    # 4. In-flight and new predictions immediately use v2 without server restart!
    r2 = await client.post("/api/v1/predict/ecg", json={"signal": fake_signal})
    assert r2.status_code == 200
    assert r2.json()["model_version"] == "v2"

    # 5. Instant zero-downtime rollback to v1
    rollback_res = await client.post("/api/models/v1/activate")
    assert rollback_res.status_code == 200
    r3 = await client.post("/api/v1/predict/ecg", json={"signal": fake_signal})
    assert r3.status_code == 200
    assert r3.json()["model_version"] == "v1"
