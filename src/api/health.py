from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

type HealthStatus = Literal["ok", "error"]

class HealthResponse(BaseModel):
    status: HealthStatus = "ok"

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")
