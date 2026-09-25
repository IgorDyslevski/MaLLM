from fastapi import FastAPI

from src.api import v1_router
from src.config import Settings


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description=settings.app.description,
    )

    app.include_router(v1_router)
    return app


app = create_app()
