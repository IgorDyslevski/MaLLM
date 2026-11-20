import os
from unittest.mock import AsyncMock

import pytest
from faststream.rabbit import RabbitBroker

os.environ.setdefault("DATABASE__USERNAME", "test-user")
os.environ.setdefault("DATABASE__PASSWORD", "test-password")


@pytest.fixture(autouse=True)
def api_broker(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    broker = AsyncMock(spec=RabbitBroker)
    broker.__aenter__.return_value = broker
    monkeypatch.setattr("src.app.create_broker", lambda settings: broker)
    return broker
