from fastapi import APIRouter

from src.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/healthz")
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")
