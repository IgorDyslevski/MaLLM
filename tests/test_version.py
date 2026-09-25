import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.app import create_app


def test_version_returns_application_version_from_project() -> None:
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    expected = tomllib.loads(project_file.read_text(encoding="utf-8"))["project"][
        "version"
    ]
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/version")
        openapi = client.get("/openapi.json").json()

    assert response.status_code == 200
    assert response.json() == {"version": expected}
    assert openapi["info"]["version"] == expected


def test_version_ignores_app_version_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP__VERSION", "incorrect-version")

    with TestClient(create_app()) as client:
        response = client.get("/api/v1/version")

    assert response.json()["version"] != "incorrect-version"
