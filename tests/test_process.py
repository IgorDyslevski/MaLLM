import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from aio_pika.exceptions import DeliveryError
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
from pamqp.commands import Basic

from src.app import create_app
from src.exceptions import InferenceError
from src.schemas import GenerationRequest
from src.services.inference import process


def test_api_returns_task_id_only_after_confirm(api_broker: AsyncMock) -> None:
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/tasks", json={"prompt": "a"})
    assert response.status_code == 202
    task = api_broker.publish.await_args.args[0]
    assert task.task_id == response.json()["task_id"]
    assert task.request == GenerationRequest(prompt="a")


def test_api_rejects_overload_after_one_publish(api_broker: AsyncMock) -> None:
    api_broker.publish.side_effect = DeliveryError(None, Basic.Nack())
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/tasks", json={"prompt": "a"})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    api_broker.publish.assert_awaited_once()


def test_api_bounds_wait_for_unresponsive_broker(
    api_broker: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.process.PUBLISH_TIMEOUT_SECONDS", 0.02)

    async def stall(*args: object, **kwargs: object) -> None:
        await asyncio.Event().wait()

    api_broker.publish.side_effect = stall
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/tasks", json={"prompt": "a"})
    assert response.status_code == 503
    assert int(response.headers["Retry-After"]) >= 1
    api_broker.publish.assert_awaited_once()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": ""},
        {"prompt": "a", "max_new_tokens": 0},
        {"prompt": "a", "max_new_tokens": 1025},
        {"prompt": "a", "seed": -1},
    ],
)
def test_api_validates_input(payload: dict, api_broker: AsyncMock) -> None:
    with TestClient(create_app()) as client:
        assert client.post("/api/v1/tasks", json=payload).status_code == 422
    api_broker.publish.assert_not_awaited()


@pytest.mark.parametrize("error", [MlflowException, RuntimeError, ValueError])
def test_inference_preserves_original_error_cause(error: type[Exception]) -> None:
    model = Mock(spec=PyFuncModel)
    cause = error("original details")
    model.predict.side_effect = cause
    with pytest.raises(InferenceError) as caught:
        process(GenerationRequest(prompt="a"), model)
    assert caught.value.__cause__ is cause
