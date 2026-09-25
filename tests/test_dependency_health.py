import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from psycopg import OperationalError
from pydantic import ValidationError

from src.api.v1.health import database_settings
from src.app import create_app
from src.config import DatabaseSettings
from src.schemas import ComponentHealth
from src.services.health import health_report, postgres_version


@pytest.fixture
def settings() -> DatabaseSettings:
    return DatabaseSettings()


@pytest.fixture
def client(settings: DatabaseSettings) -> TestClient:
    app = create_app()
    app.dependency_overrides[database_settings] = lambda: settings
    return TestClient(app)


def test_health_reports_server_version_and_response_time(
    client: TestClient, settings: DatabaseSettings
) -> None:
    probe = AsyncMock(return_value="16.4")
    with patch("src.services.health.postgres_version", probe):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "components": {
            "postgres": {
                "status": "ok",
                "message": "PostgreSQL is available",
                "version": "16.4",
                "response_time_ms": response.json()["components"]["postgres"][
                    "response_time_ms"
                ],
            }
        },
    }
    assert response.json()["components"]["postgres"]["response_time_ms"] >= 0
    probe.assert_awaited_once_with(settings)


@pytest.mark.parametrize("error", [OperationalError, OSError, TimeoutError, ValueError])
def test_health_degraded_when_postgres_unavailable(
    client: TestClient, error: type[Exception]
) -> None:
    probe = AsyncMock(side_effect=error("secret connection details"))
    with patch("src.services.health.postgres_version", probe):
        response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    component = response.json()["components"]["postgres"]
    assert component["status"] == "error"
    assert component["version"] is None
    assert component["response_time_ms"] >= 0
    assert component["message"] == "PostgreSQL is unavailable"
    assert "secret" not in response.text


def test_health_report_limits_probe_to_three_seconds(
    settings: DatabaseSettings,
) -> None:
    with patch(
        "src.services.health.asyncio.wait_for", side_effect=TimeoutError
    ) as wait:
        report = asyncio.run(health_report(settings))

    wait.assert_called_once()
    assert wait.call_args.kwargs == {"timeout": 3}
    wait.call_args.args[0].close()
    assert report.status == "degraded"


def test_health_report_measures_complete_probe_duration(
    settings: DatabaseSettings,
) -> None:
    with (
        patch("src.services.health.postgres_version", new_callable=AsyncMock) as probe,
        patch("src.services.health.perf_counter", side_effect=[10.0, 10.125]),
    ):
        probe.return_value = "16.4"
        report = asyncio.run(health_report(settings))

    assert report.components["postgres"].response_time_ms == 125.0


def test_postgres_version_queries_running_server_and_closes_connection(
    settings: DatabaseSettings,
) -> None:
    connection = AsyncMock()
    connection.__aenter__.return_value = connection
    connection.execute.return_value.fetchone.return_value = ("16.4",)
    with patch(
        "src.services.health.AsyncConnection.connect", return_value=connection
    ) as connect:
        assert asyncio.run(postgres_version(settings)) == "16.4"

    connect.assert_awaited_once_with(
        host=settings.host,
        port=settings.port,
        user=settings.username,
        password=settings.password,
        dbname=settings.database_name,
        connect_timeout=3,
    )
    connection.execute.assert_awaited_once_with("SHOW server_version")
    connection.__aexit__.assert_awaited_once()


def test_postgres_version_rejects_missing_server_version(
    settings: DatabaseSettings,
) -> None:
    connection = AsyncMock()
    connection.__aenter__.return_value = connection
    connection.execute.return_value.fetchone.return_value = None
    with patch("src.services.health.AsyncConnection.connect", return_value=connection):
        with pytest.raises(ValueError, match="did not return a server version"):
            asyncio.run(postgres_version(settings))


def test_component_health_rejects_negative_response_time() -> None:
    with pytest.raises(ValidationError):
        ComponentHealth(status="ok", message="available", response_time_ms=-1)
