from fastapi.testclient import TestClient

from bile_scripture.serving.api import app


def test_health() -> None:
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True
