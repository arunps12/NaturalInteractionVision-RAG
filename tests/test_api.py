"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from visionllm_interactionanalysis.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "gpu_available" in data


def test_predict_no_checkpoint(client):
    """Predict should return 503 when no checkpoint is configured."""
    import io
    resp = client.post(
        "/predict",
        files={"file": ("test.jpg", io.BytesIO(b"\xff\xd8\xff\xe0"), "image/jpeg")},
    )
    assert resp.status_code == 503
