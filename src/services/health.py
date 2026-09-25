import asyncio
import logging
from time import perf_counter

from psycopg import AsyncConnection, Error

from src.config import DatabaseSettings
from src.schemas import ComponentHealth, DependencyHealthResponse

logger = logging.getLogger(__name__)


async def postgres_version(settings: DatabaseSettings) -> str:
    """Получить версию PostgreSQL с работающего сервера."""
    async with await AsyncConnection.connect(
        host=settings.host,
        port=settings.port,
        user=settings.username,
        password=settings.password,
        dbname=settings.database_name,
        connect_timeout=3,
    ) as connection:
        cursor = await connection.execute("SHOW server_version")
        row = await cursor.fetchone()
        if row is None:
            raise ValueError("PostgreSQL did not return a server version")
        return str(row[0])


async def health_report(settings: DatabaseSettings) -> DependencyHealthResponse:
    """Проверить PostgreSQL и измерить время полного обращения."""
    started = perf_counter()
    try:
        version = await asyncio.wait_for(postgres_version(settings), timeout=3)
    except (Error, OSError, TimeoutError, ValueError):
        logger.exception("PostgreSQL health check failed")
        component = ComponentHealth(
            status="error",
            message="PostgreSQL is unavailable",
            response_time_ms=(perf_counter() - started) * 1000,
        )
    else:
        component = ComponentHealth(
            status="ok",
            message="PostgreSQL is available",
            version=version,
            response_time_ms=(perf_counter() - started) * 1000,
        )
    return DependencyHealthResponse(
        status="ok" if component.status == "ok" else "degraded",
        components={"postgres": component},
    )
