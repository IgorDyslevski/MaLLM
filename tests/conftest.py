import os
from unittest.mock import Mock

import pytest
from mlflow.pyfunc import PyFuncModel

os.environ.setdefault("DATABASE__USERNAME", "test-user")
os.environ.setdefault("DATABASE__PASSWORD", "test-password")


@pytest.fixture(autouse=True)
def inference_loader(monkeypatch: pytest.MonkeyPatch) -> Mock:
    loader = Mock(return_value=Mock(spec=PyFuncModel))
    monkeypatch.setattr("src.app.load_registered_model", loader)
    return loader
