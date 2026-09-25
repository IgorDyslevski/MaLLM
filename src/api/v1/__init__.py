from fastapi import APIRouter

from src.api.v1.version import version_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(version_router)
