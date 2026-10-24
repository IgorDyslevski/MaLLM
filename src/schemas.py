from pathlib import Path
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


class DatasetRecord(BaseModel):
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DatasetSpec(BaseModel):
    name: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source: str = Field(pattern=r"^https://huggingface.co/datasets/")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_rows: int = Field(gt=0)
    split: Literal["train"] = "train"
    path: Path


class Dataset(BaseModel):
    spec: DatasetSpec
    records: list[DatasetRecord] = Field(min_length=1)


class DatasetTrackingResult(BaseModel):
    run_id: str
    experiment_id: str
    dataset_digest: str
    artifact_uri: str
