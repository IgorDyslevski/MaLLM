from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class VersionResponse(BaseModel):
    version: str = Field(min_length=1)
