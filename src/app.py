import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient
from starlette.concurrency import run_in_threadpool

from src.api import api_router
from src.config import Settings
from src.exceptions import InferenceError
from src.logs import log_requests, setup_logging
from src.schemas import InferenceErrorResponse
from src.services.model_registry import load_registered_model
from src.services.version import get_version


def create_app() -> FastAPI:
    settings = Settings()
    setup_logging(settings.log_level)
    version = get_version(Path(__file__).resolve().parents[1] / "pyproject.toml")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[dict[str, PyFuncModel]]:
        client = MlflowClient(
            tracking_uri=settings.inference.tracking_uri,
            registry_uri=settings.inference.tracking_uri,
        )
        model = await run_in_threadpool(
            load_registered_model, client, settings.inference.model_uri
        )
        yield {"model": model}

    app = FastAPI(
        title=settings.app.name,
        version=version.version,
        description=settings.app.description,
        lifespan=lifespan,
    )

    @app.exception_handler(InferenceError)
    async def inference_error(request: Request, exc: InferenceError) -> JSONResponse:
        logging.getLogger(__name__).error("Model inference failed", exc_info=exc)
        return JSONResponse(
            status_code=503, content=InferenceErrorResponse().model_dump()
        )

    app.middleware("http")(log_requests)
    app.include_router(api_router)
    return app


app = create_app()
