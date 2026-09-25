import json
import logging
import sys

import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.logs import JsonFormatter


def test_request_id_is_taken_from_header() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/healthz", headers={"X-Request-ID": "abc-123"})

    assert response.headers["X-Request-ID"] == "abc-123"


def test_request_id_is_generated_when_header_missing() -> None:
    with TestClient(create_app()) as client:
        first = client.get("/healthz").headers["X-Request-ID"]
        second = client.get("/healthz").headers["X-Request-ID"]

    assert len(first) == 32
    assert first != second


def test_request_completed_is_logged_with_request_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with TestClient(create_app()) as client:
        client.get("/healthz", headers={"X-Request-ID": "abc-123"})

    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    line = next(line for line in lines if line["message"] == "request completed")
    assert line["request_id"] == "abc-123"
    assert line["method"] == "GET"
    assert line["path"] == "/healthz"
    assert line["status_code"] == 200
    assert line["duration_ms"] >= 0


def test_json_formatter_includes_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.getLogger("test").makeRecord(
            "test", logging.ERROR, __file__, 1, "failed", None, sys.exc_info()
        )

    line = json.loads(JsonFormatter().format(record))
    assert line["message"] == "failed"
    assert line["level"] == "ERROR"
    assert "ValueError: boom" in line["exception"]
