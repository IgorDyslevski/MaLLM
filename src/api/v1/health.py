from typing import Annotated

from fastapi import APIRouter, Depends, Response

from src.config import DatabaseSettings, Settings
from src.schemas import DependencyHealthResponse
from src.services.health import health_report

health_router = APIRouter(tags=["Health"])


def database_settings() -> DatabaseSettings:
    """Вернуть настройки PostgreSQL из конфигурации приложения."""
    return Settings().database


@health_router.get("/health", responses={503: {"model": DependencyHealthResponse}})
async def health_check(
    response: Response,
    settings: Annotated[DatabaseSettings, Depends(database_settings)],
) -> DependencyHealthResponse:
    report = await health_report(settings)
    response.status_code = 200 if report.status == "ok" else 503
    return report
