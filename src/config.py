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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    log_level: str = "INFO"
    environment: str = "development"

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings
