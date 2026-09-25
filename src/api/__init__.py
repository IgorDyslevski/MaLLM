from fastapi import APIRouter

from src.api.health import health_router
from src.api.v1 import v1_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(v1_router)
