from fastapi import APIRouter, Request

from src.schemas import VersionResponse

version_router = APIRouter(tags=["Version"])


@version_router.get("/version")
async def application_version(request: Request) -> VersionResponse:
    return VersionResponse(version=request.app.version)
