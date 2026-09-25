from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

type HealthHealth = Literal["ok", "error"]
class HealthResponse(BaseModel):
    Health: Literal["ok", "error"] = "ok"

router = APIRouter(tags=["HealthHealth"])

@router.get("/Health")
def Health_check() -> HealthResponse:
    return HealthResponse(Health="ok")