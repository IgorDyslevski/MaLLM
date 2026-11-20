from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from faststream.rabbit import RabbitBroker

from src.api import api_router
from src.config import PipelineSettings, Settings
from src.logs import log_requests, setup_logging
from src.services.pipeline import create_broker, task_queue
from src.services.version import get_version


def create_app() -> FastAPI:
    settings = Settings()
    setup_logging(settings.log_level, "api", settings.environment)
    version = get_version(Path(__file__).resolve().parents[1] / "pyproject.toml")

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[dict[str, RabbitBroker | PipelineSettings]]:
        pipeline = PipelineSettings()
        async with create_broker(pipeline) as broker:
            await broker.declare_queue(task_queue(pipeline))
            yield {"broker": broker, "pipeline": pipeline}

    app = FastAPI(
        title=settings.app.name,
        version=version.version,
        description=settings.app.description,
        lifespan=lifespan,
    )

    app.middleware("http")(log_requests)
    app.include_router(api_router)
    return app


app = create_app()
