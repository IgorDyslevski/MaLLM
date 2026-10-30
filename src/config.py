from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

type Environment = Literal["development", "staging", "production"]


class AppSettings(BaseModel):
    name: str = "Mallm"
    description: str = "Mathematical large language model from scratch"


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    username: str
    password: str
    database_name: str = "mallm_db"


class InferenceSettings(BaseModel):
    tracking_uri: str = Field(default="http://127.0.0.1:5000", min_length=1)
    model_uri: str = Field(default="models:/MaLLM@champion", min_length=1)


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_", env_file=".env", extra="ignore"
    )

    rabbit_url: str = Field(default="amqp://guest:guest@127.0.0.1:5672/", repr=False)
    queue: str = Field(default="mallm.inference", min_length=1)
    queue_capacity: int = Field(default=50, gt=0)
    retry_seconds: float = Field(default=0.1, gt=0)
    source_interval_seconds: float = Field(default=0, ge=0)
    inference_seconds: float = Field(default=0.2, gt=0)
    tracking_uri: str = Field(default="http://127.0.0.1:5000", min_length=1)
    model_uri: str = Field(
        default="models:/MaLLM@champion", pattern=r"^models:/[^/@]+@[^/@]+$"
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    log_level: str = "INFO"
    environment: str = "development"

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
