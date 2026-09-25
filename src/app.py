from fastapi import FastAPI, APIRouter

from src.config import Settings
from src.api import v1_router

router = APIRouter()
@router.get("/health")
def health_check() -> dict[str, str]:
    return {"health": "ok"}

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
