import tomllib
from pathlib import Path

from src.schemas import VersionResponse


def get_version(project_file: Path) -> VersionResponse:
    project = tomllib.loads(project_file.read_text(encoding="utf-8"))
    return VersionResponse.model_validate(project["project"])
