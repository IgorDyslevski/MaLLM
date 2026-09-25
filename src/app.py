from fastapi import FastAPI

from src.api.health import router
from src.config import Settings


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description=settings.app.description,
    )

    app.include_router(router)
    return app


app = create_app()
