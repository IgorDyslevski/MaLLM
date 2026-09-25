from fastapi import APIRouter
from .health import router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(router)

__all__ = {"router",}
