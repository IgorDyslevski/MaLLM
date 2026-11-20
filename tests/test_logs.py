import asyncio
import json
import logging
import sys

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.app import create_app
from src.logs import JsonFormatter, setup_logging, task_context, task_id
from src.schemas import InferenceTask


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
        client.get("/api/v1/version", headers={"X-Request-ID": "abc-123"})

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    line = next(line for line in lines if line["message"] == "request completed")
    assert line["request_id"] == "abc-123"
    assert line["method"] == "GET"
    assert line["path"] == "/api/v1/version"
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


def test_health_checks_do_not_log_request_completion_at_info(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with TestClient(create_app()) as client:
        client.get("/healthz")
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert not any(line["message"] == "request completed" for line in lines)


def test_library_logs_use_json_stdout_and_configured_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    library = logging.getLogger("library-test")
    library.addHandler(logging.StreamHandler())
    library.setLevel(logging.INFO)
    setup_logging("WARNING", "worker", "staging")
    library.info("hidden")
    library.warning("visible")
    output = capsys.readouterr()
    line = json.loads(output.out)
    assert output.err == ""
    assert line["message"] == "visible"
    assert line["service"] == "worker"
    assert line["env"] == "staging"
    assert line["level"] == "WARNING"


def test_task_context_is_isolated_between_tasks_and_inherited_by_threads() -> None:
    async def inspect(identifier: str) -> str | None:
        with task_context(identifier):
            await asyncio.sleep(0)
            return await asyncio.to_thread(task_id.get)

    async def scenario() -> None:
        assert await asyncio.gather(inspect("one"), inspect("two")) == ["one", "two"]
        assert task_id.get() is None

    asyncio.run(scenario())


def test_task_context_is_restored_after_error() -> None:
    with task_context("outer"):
        with pytest.raises(ValueError), task_context("inner"):
            raise ValueError("failed")
        assert task_id.get() == "outer"
    assert task_id.get() is None


def test_formatter_adds_task_id_without_manual_fields() -> None:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "event", (), None)
    with task_context("batch:1"):
        assert json.loads(JsonFormatter().format(record))["task_id"] == "batch:1"
    assert json.loads(JsonFormatter().format(record))["task_id"] is None


def test_formatter_correlates_framework_logs_by_full_message_id() -> None:
    record = logging.LogRecord(
        "faststream.rabbit", logging.INFO, __file__, 1, "Received", (), None
    )
    record.message_id = "batch:12345678901234567890"
    line = json.loads(JsonFormatter().format(record))
    assert line["task_id"] == record.message_id


def test_formatter_redacts_dsn_and_configured_secrets() -> None:
    record = logging.LogRecord(
        "test",
        logging.ERROR,
        __file__,
        1,
        "amqp://guest:private@rabbit/ password=hidden credential=секрет",
        (),
        None,
    )
    rendered = JsonFormatter(secrets=("секрет",)).format(record)
    assert "private" not in rendered
    assert "hidden" not in rendered
    assert "секрет" not in rendered
    assert "[REDACTED]" in json.loads(rendered)["message"]


def test_numeric_secret_redaction_preserves_valid_json_fields() -> None:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "secret=1", (), None)
    record.fields = {"attempt": 1}
    line = json.loads(JsonFormatter(secrets=("1",)).format(record))
    assert line["attempt"] == 1
    assert "[REDACTED]" in line["message"]


def test_validation_stacktrace_does_not_expose_task_payload() -> None:
    body = b'"private-file-record"'
    try:
        InferenceTask.model_validate_json(body)
    except ValidationError:
        record = logging.LogRecord(
            "test", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
        )
    rendered = JsonFormatter().format(record)
    assert "private-file-record" not in rendered
    assert "ValidationError" in json.loads(rendered)["exception"]
