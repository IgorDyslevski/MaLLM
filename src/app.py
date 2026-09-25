from pathlib import Path

from fastapi import FastAPI

from src.api import api_router
from src.config import Settings
from src.logs import log_requests, setup_logging
from src.services.version import get_version


def create_app() -> FastAPI:
    settings = Settings()
    setup_logging(settings.log_level)
    version = get_version(Path(__file__).resolve().parents[1] / "pyproject.toml")
    app = FastAPI(
        title=settings.app.name,
        version=version.version,
        description=settings.app.description,
    )

    app.middleware("http")(log_requests)
    app.include_router(api_router)
    return app


app = create_app()
