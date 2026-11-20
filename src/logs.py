import json
import logging
import os
import re
import sys
import traceback
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from time import perf_counter
from types import TracebackType
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ValidationError

logger = logging.getLogger(__name__)

request_id: ContextVar[str] = ContextVar("request_id", default="-")
task_id: ContextVar[str | None] = ContextVar("task_id", default=None)


@contextmanager
def task_context(identifier: str) -> Iterator[None]:
    """Связать логи текущей задачи и восстановить предыдущий контекст."""
    token = task_id.set(identifier)
    try:
        yield
    finally:
        task_id.reset(token)


class JsonFormatter(logging.Formatter):
    """Форматировать запись лога в одну JSON-строку."""

    def __init__(
        self,
        service: str = "api",
        env: str = "development",
        secrets: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.service = service
        self.env = env
        self.secrets = secrets

    def formatException(  # noqa: N802
        self,
        exc_info: tuple[
            type[BaseException] | None, BaseException | None, TracebackType | None
        ],
    ) -> str:
        """Сохранить стек без входных данных из ошибок Pydantic."""
        error = exc_info[1]
        if error is None:
            return ""
        cause = error.__cause__ or (
            error.__context__ if not error.__suppress_context__ else None
        )
        prefix = ""
        if cause is not None:
            prefix = (
                self.formatException((type(cause), cause, cause.__traceback__))
                + "\nThe above exception caused the following exception:\n"
            )
        if isinstance(error, ValidationError):
            return (
                prefix
                + "".join(traceback.format_tb(exc_info[2]))
                + (f"ValidationError: {error.error_count()} validation error(s)")
            )
        return prefix + "".join(traceback.format_exception(*exc_info, chain=False))

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id.get(),
            **getattr(record, "fields", {}),
            "service": self.service,
            "env": self.env,
            "task_id": task_id.get() or getattr(record, "message_id", None),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(
            {
                key: self.redact(value) if isinstance(value, str) else value
                for key, value in payload.items()
            },
            ensure_ascii=False,
            default=str,
        )

    def redact(self, value: str) -> str:
        """Скрыть учётные данные перед сериализацией JSON."""
        rendered = re.sub(r"(\w+://[^\s/@:]+:)[^\s/@]+@", r"\1[REDACTED]@", value)
        rendered = re.sub(
            r"(?i)((?:password|passwd|token|secret|api_key)[= :]+)[^\s,;\"\\]+",
            r"\1[REDACTED]",
            rendered,
        )
        for secret in self.secrets:
            rendered = rendered.replace(secret, "[REDACTED]")
        return rendered


def setup_logging(level: str, service: str = "api", env: str = "development") -> None:
    """Направить логи приложения и uvicorn в stdout в формате JSON."""
    secrets = tuple(
        value
        for key, value in os.environ.items()
        if value
        and any(part in key.upper() for part in ("PASSWORD", "SECRET", "TOKEN"))
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter(service, env, secrets))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    logging.captureWarnings(True)
    for entry in logging.root.manager.loggerDict.values():
        if isinstance(entry, logging.Logger):
            entry.handlers = []
            entry.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Присвоить запросу request_id и залогировать итог его обработки."""
    token = request_id.set(request.headers.get("X-Request-ID") or uuid4().hex)
    started = perf_counter()
    try:
        response = await call_next(request)
    finally:
        identifier = request_id.get()
        request_id.reset(token)
    response.headers["X-Request-ID"] = identifier
    level = (
        logging.DEBUG
        if request.url.path in ("/healthz", "/api/v1/health")
        else logging.INFO
    )
    logger.log(
        level,
        "request completed",
        extra={
            "fields": {
                "request_id": identifier,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            }
        },
    )
    return response
