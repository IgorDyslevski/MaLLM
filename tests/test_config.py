import pytest
from pydantic import ValidationError

from src.config import Settings


def test_settings_require_database_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE__USERNAME")
    monkeypatch.delenv("DATABASE__PASSWORD")

    with pytest.raises(ValidationError):
        Settings()
