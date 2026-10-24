import asyncio
import threading
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel

from src.api.process import process_text
from src.app import create_app
from src.config import Settings
from src.exceptions import InferenceError
from src.schemas import GenerationRequest, GenerationResult
from src.services.inference import process


def test_process_reuses_model_loaded_once_at_startup(inference_loader: Mock) -> None:
    model = inference_loader.return_value
    model.predict.return_value = pd.DataFrame([{"text": "2 + 2 = 4"}])
    with TestClient(create_app()) as client:
        inference_loader.assert_called_once()
        assert model.predict.call_count == 0
        for _ in range(2):
            response = client.post(
                "/process", json={"prompt": "2 + 2 =", "max_new_tokens": 2, "seed": 7}
            )
            assert response.status_code == 200
            assert response.json() == {"text": "2 + 2 = 4"}
        assert client.get("/healthz").status_code == 200
    inference_loader.assert_called_once()
    assert model.predict.call_count == 2
    assert model.predict.call_args.args[0].to_dict("records") == [
        {"prompt": "2 + 2 =", "max_new_tokens": 2, "seed": 7}
    ]


def test_process_uses_default_generation_parameters(inference_loader: Mock) -> None:
    model = inference_loader.return_value
    model.predict.return_value = pd.DataFrame([{"text": "hello"}])
    with TestClient(create_app()) as client:
        assert client.post("/process", json={"prompt": "hi"}).status_code == 200
    assert model.predict.call_args.args[0].to_dict("records") == [
        {"prompt": "hi", "max_new_tokens": 32, "seed": 42}
    ]


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
def test_process_rejects_invalid_input_without_inference(
    payload: dict[str, str | int],
    inference_loader: Mock,
) -> None:
    with TestClient(create_app()) as client:
        assert client.post("/process", json=payload).status_code == 422
    inference_loader.return_value.predict.assert_not_called()


def test_process_hides_model_error_details(inference_loader: Mock) -> None:
    inference_loader.return_value.predict.side_effect = RuntimeError("secret path")
    with TestClient(create_app()) as client:
        response = client.post("/process", json={"prompt": "a"})
    assert response.status_code == 503
    assert response.json() == {
        "status": "inference_error",
        "message": "Model inference failed",
    }
    assert "secret" not in response.text


def test_process_runs_inference_outside_event_loop() -> None:
    model = Mock(spec=PyFuncModel)
    caller = threading.get_ident()
    worker_threads: list[int] = []

    def predict(inputs: pd.DataFrame) -> pd.DataFrame:
        worker_threads.append(threading.get_ident())
        return pd.DataFrame([{"text": inputs.iloc[0]["prompt"] + "b"}])

    model.predict.side_effect = predict
    result = asyncio.run(process_text(GenerationRequest(prompt="a"), model))
    assert result == GenerationResult(text="ab")
    assert worker_threads and worker_threads[0] != caller


def test_startup_fails_when_model_cannot_be_loaded(inference_loader: Mock) -> None:
    inference_loader.side_effect = MlflowException("Registry unavailable")
    with pytest.raises(MlflowException, match="Registry unavailable"):
        with TestClient(create_app()):
            pytest.fail("Startup should fail")
    inference_loader.assert_called_once()


def test_startup_uses_configured_registry_and_model(
    monkeypatch: pytest.MonkeyPatch,
    inference_loader: Mock,
) -> None:
    monkeypatch.setenv("INFERENCE__TRACKING_URI", "http://registry.test:5000")
    monkeypatch.setenv("INFERENCE__MODEL_URI", "models:/MaLLM/2")
    with TestClient(create_app()):
        client, uri = inference_loader.call_args.args
        assert client.tracking_uri == "http://registry.test:5000"
        assert uri == "models:/MaLLM/2"
    assert Settings().inference.model_uri == "models:/MaLLM/2"


@pytest.mark.parametrize("error", [MlflowException, RuntimeError, ValueError])
def test_inference_preserves_original_error_cause(error: type[Exception]) -> None:
    model = Mock(spec=PyFuncModel)
    cause = error("original details")
    model.predict.side_effect = cause
    with pytest.raises(InferenceError) as caught:
        process(GenerationRequest(prompt="a"), model)
    assert caught.value.__cause__ is cause
