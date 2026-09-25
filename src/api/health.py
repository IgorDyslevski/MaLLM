from fastapi import APIRouter

from src.schemas import HealthResponse

health_router = APIRouter(tags=["Health"])


@health_router.get("/healthz")
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")
