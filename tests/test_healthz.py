from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.health import health_check
from src.app import create_app
from src.schemas import HealthResponse


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


def test_healthz_returns_ok_when_application_is_running(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"status": "ok"}


def test_healthz_rejects_post_request(client: TestClient) -> None:
    response = client.post("/healthz")

    assert response.status_code == 405


def test_health_check_returns_health_response() -> None:
    assert health_check() == HealthResponse(status="ok")


def test_health_response_defaults_to_ok() -> None:
    assert HealthResponse().model_dump() == {"status": "ok"}


def test_health_response_rejects_error_status() -> None:
    with pytest.raises(ValidationError):
        HealthResponse.model_validate({"status": "error"})
