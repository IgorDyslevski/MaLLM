import asyncio
import logging
from time import perf_counter

from psycopg import AsyncConnection, Error

from src.config import DatabaseSettings
from src.schemas import ComponentHealth, DependencyHealthResponse

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 3


async def postgres_version(settings: DatabaseSettings) -> str:
    """Получить версию PostgreSQL запросом к работающему серверу."""
    async with await AsyncConnection.connect(
        host=settings.host,
        port=settings.port,
        user=settings.username,
        password=settings.password,
        dbname=settings.database_name,
    ) as connection:
        cursor = await connection.execute("SHOW server_version")
        row = await cursor.fetchone()
    assert row is not None
    return str(row[0])


async def health_report(settings: DatabaseSettings) -> DependencyHealthResponse:
    """Проверить PostgreSQL и измерить время полного обращения."""
    started = perf_counter()
    version: str | None = None
    try:
        version = await asyncio.wait_for(
            postgres_version(settings), timeout=PROBE_TIMEOUT_SECONDS
        )
    except (Error, TimeoutError):
        logger.exception("PostgreSQL health check failed")
    available = version is not None
    component = ComponentHealth(
        status="ok" if available else "error",
        message="PostgreSQL is available" if available else "PostgreSQL is unavailable",
        version=version,
        response_time_ms=round((perf_counter() - started) * 1000, 2),
    )
    return DependencyHealthResponse(
        status="ok" if available else "degraded",
        components={"postgres": component},
    )
