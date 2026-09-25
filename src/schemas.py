from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class VersionResponse(BaseModel):
    version: str = Field(min_length=1)


class ComponentHealth(BaseModel):
    status: Literal["ok", "error"]
    message: str
    version: str | None = None
    response_time_ms: float = Field(ge=0)


class DependencyHealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: dict[str, ComponentHealth]
