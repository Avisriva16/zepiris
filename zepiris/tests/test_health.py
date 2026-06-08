"""Health routes without full app lifespan (no MinIO/Milvus)."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from zepiris.api.routes.health import router as health_router


def test_healthz_returns_ok() -> None:
    app = FastAPI()
    app.include_router(health_router)
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
