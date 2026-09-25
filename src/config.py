from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

type Environment = Literal["development", "staging", "production"]


class AppSettings(BaseModel):
    name: str = "Mallm"
    version: str = "0.1.0"
    description: str = "Mathematical large language model from scratch"


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    username: str = "user"
    password: str = "password"
    database_name: str = "mallm_db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file="config.yaml", env_nested_delimiter="__"
    )

    log_level: str = "INFO"
    environment: str = "development"

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls))
