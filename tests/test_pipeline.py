import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from aio_pika.exceptions import DeliveryError
from faststream.rabbit import RabbitBroker, TestRabbitBroker
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient
from pamqp.commands import Basic
from pydantic import ValidationError

from src.config import PipelineSettings
from src.schemas import (
    GenerationRequest,
    GenerationResult,
    InferenceErrorResponse,
    InferenceTask,
    TaskResult,
)
from src.services.pipeline import (
    append_result,
    create_worker,
    execute_task,
    initialize_worker,
    publish_file,
    publish_task,
    task_queue,
)


def test_pipeline_settings_require_model_alias() -> None:
    with pytest.raises(ValidationError):
        PipelineSettings(model_uri="models:/MaLLM/1", _env_file=None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("queue_capacity", 0),
        ("inference_seconds", 0),
        ("source_interval_seconds", -1),
        ("retry_seconds", 0),
    ],
)
def test_pipeline_settings_reject_invalid_limits(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        PipelineSettings.model_validate({field: value})


def test_task_queue_limits_ready_messages_without_dropping_old_tasks() -> None:
    queue = task_queue(PipelineSettings(queue_capacity=7, _env_file=None))
    assert queue.durable
    assert queue.arguments["x-max-length"] == 7
    assert queue.arguments["x-overflow"] == "reject-publish"


def test_publish_task_retries_same_message_when_queue_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = Mock(spec=RabbitBroker)
    broker.publish = AsyncMock(side_effect=[DeliveryError(None, Basic.Nack()), None])
    sleep = AsyncMock()
    monkeypatch.setattr("src.services.pipeline.asyncio.sleep", sleep)
    task = InferenceTask(task_id="batch:1", request=GenerationRequest(prompt="a"))
    settings = PipelineSettings(_env_file=None)
    asyncio.run(publish_task(task, broker, settings))
    assert broker.publish.await_count == 2
    assert broker.publish.await_args_list[0] == broker.publish.await_args_list[1]
    assert broker.publish.await_args.args == (task,)
    assert broker.publish.await_args.kwargs["persist"] is True
    sleep.assert_awaited_once_with(settings.retry_seconds)


def test_publish_task_propagates_connection_failure() -> None:
    broker = Mock(spec=RabbitBroker)
    broker.publish = AsyncMock(side_effect=ConnectionError("offline"))
    task = InferenceTask(task_id="batch:1", request=GenerationRequest(prompt="a"))
    with pytest.raises(ConnectionError, match="offline"):
        asyncio.run(publish_task(task, broker, PipelineSettings(_env_file=None)))
    broker.publish.assert_awaited_once()


def test_publish_task_does_not_retry_unroutable_message() -> None:
    broker = Mock(spec=RabbitBroker)
    cause = DeliveryError(None, Basic.Return())
    broker.publish = AsyncMock(side_effect=cause)
    task = InferenceTask(task_id="batch:1", request=GenerationRequest(prompt="a"))
    with pytest.raises(RuntimeError, match="could not route") as caught:
        asyncio.run(publish_task(task, broker, PipelineSettings(_env_file=None)))
    assert caught.value.__cause__ is cause
    broker.publish.assert_awaited_once()


def test_publish_file_streams_validated_records_with_stable_ids(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text('{"prompt":"a"}\n{"prompt":"b","seed":7}\n')
    broker = Mock(spec=RabbitBroker)
    result = asyncio.run(
        publish_file(source, "batch", broker, PipelineSettings(_env_file=None))
    )
    assert result.published == 2
    assert result.run_id == "batch"
    assert result.elapsed_seconds >= 0
    tasks = [call.args[0] for call in broker.publish.await_args_list]
    assert [task.task_id for task in tasks] == ["batch:1", "batch:2"]
    assert tasks[1].request.seed == 7
    broker.declare_queue.assert_awaited_once()


def test_publish_file_waits_for_publication_before_reading_next_record(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text('{"prompt":"a"}\ninvalid JSON\n')
    broker = Mock(spec=RabbitBroker)
    broker.publish = AsyncMock(side_effect=ConnectionError("offline"))
    with pytest.raises(ConnectionError, match="offline"):
        asyncio.run(
            publish_file(source, "batch", broker, PipelineSettings(_env_file=None))
        )
    broker.publish.assert_awaited_once()


def test_publish_file_rejects_invalid_record_without_publishing(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text('{"prompt":""}\n')
    broker = Mock(spec=RabbitBroker)
    with pytest.raises(ValidationError):
        asyncio.run(
            publish_file(source, "batch", broker, PipelineSettings(_env_file=None))
        )
    broker.publish.assert_not_awaited()


def test_execute_task_runs_real_prediction_outside_event_loop() -> None:
    caller_thread = threading.get_ident()
    worker_threads: list[int] = []
    model = Mock(spec=PyFuncModel)

    def predict(inputs: pd.DataFrame) -> pd.DataFrame:
        worker_threads.append(threading.get_ident())
        return pd.DataFrame([{"text": inputs.iloc[0]["prompt"] + "b"}])

    model.predict.side_effect = predict
    task = InferenceTask(task_id="batch:1", request=GenerationRequest(prompt="a"))
    result = asyncio.run(execute_task(task, model, 0.02))
    assert result.outcome == GenerationResult(text="ab")
    assert result.processing_seconds >= 0.02
    assert worker_threads and worker_threads[0] != caller_thread


def test_execute_task_sleeps_only_remaining_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.services.pipeline.monotonic", Mock(side_effect=[10.0, 10.3, 11.0])
    )
    sleep = AsyncMock()
    monkeypatch.setattr("src.services.pipeline.asyncio.sleep", sleep)
    model = Mock(spec=PyFuncModel)
    model.predict.return_value = pd.DataFrame([{"text": "b"}])
    task = InferenceTask(task_id="batch:1", request=GenerationRequest(prompt="a"))
    result = asyncio.run(execute_task(task, model, 1.0))
    sleep.assert_awaited_once()
    assert sleep.await_args.args[0] == pytest.approx(0.7)
    assert result.processing_seconds == 1.0


def test_execute_task_does_not_add_delay_when_inference_exceeds_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.services.pipeline.monotonic", Mock(side_effect=[10.0, 12.0, 12.0])
    )
    sleep = AsyncMock()
    monkeypatch.setattr("src.services.pipeline.asyncio.sleep", sleep)
    model = Mock(spec=PyFuncModel)
    model.predict.return_value = pd.DataFrame([{"text": "b"}])
    task = InferenceTask(task_id="batch:1", request=GenerationRequest(prompt="a"))
    asyncio.run(execute_task(task, model, 1.0))
    sleep.assert_awaited_once_with(0)


def test_execute_task_returns_stable_error_without_technical_details() -> None:
    model = Mock(spec=PyFuncModel)
    model.predict.side_effect = RuntimeError("secret model path")
    task = InferenceTask(task_id="batch:1", request=GenerationRequest(prompt="a"))
    result = asyncio.run(execute_task(task, model, 0.001))
    assert result.outcome == InferenceErrorResponse()
    assert "secret" not in result.model_dump_json()


def test_append_result_keeps_previous_results(tmp_path: Path) -> None:
    destination = tmp_path / "results.jsonl"
    result = TaskResult(
        task_id="batch:1", outcome=GenerationResult(text="b"), processing_seconds=1
    )
    append_result(destination, result)
    append_result(destination, result)
    assert [
        TaskResult.model_validate_json(line)
        for line in destination.read_text().splitlines()
    ] == [result, result]


def test_worker_loads_alias_once_and_reuses_model_for_multiple_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = PipelineSettings(inference_seconds=0.001, _env_file=None)
    client = Mock(spec=MlflowClient)
    model = Mock(spec=PyFuncModel)
    model.predict.return_value = pd.DataFrame([{"text": "b"}])
    loader = Mock(return_value=model)
    monkeypatch.setattr("src.services.pipeline.load_registered_model", loader)
    destination = tmp_path / "nested" / "results.jsonl"

    async def scenario() -> None:
        app = await initialize_worker(settings, client, destination)
        loader.assert_called_once_with(client, settings.model_uri)
        async with TestRabbitBroker(app.broker) as broker:
            for number in range(2):
                await broker.publish(
                    InferenceTask(
                        task_id=f"batch:{number}", request=GenerationRequest(prompt="a")
                    ),
                    queue=task_queue(settings),
                )

    asyncio.run(scenario())
    loader.assert_called_once()
    assert model.predict.call_count == 2
    results = [
        TaskResult.model_validate_json(line)
        for line in destination.read_text().splitlines()
    ]
    assert [result.task_id for result in results] == ["batch:0", "batch:1"]


def test_worker_startup_fails_before_consuming_when_model_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = Mock(side_effect=RuntimeError("registry unavailable"))
    monkeypatch.setattr("src.services.pipeline.load_registered_model", loader)
    destination = tmp_path / "results.jsonl"
    with pytest.raises(RuntimeError, match="registry unavailable"):
        asyncio.run(
            initialize_worker(
                PipelineSettings(_env_file=None), Mock(spec=MlflowClient), destination
            )
        )
    assert not destination.exists()


def test_worker_propagates_result_storage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = PipelineSettings(inference_seconds=0.001, _env_file=None)
    model = Mock(spec=PyFuncModel)
    model.predict.return_value = pd.DataFrame([{"text": "b"}])
    monkeypatch.setattr(
        "src.services.pipeline.append_result", Mock(side_effect=OSError("disk full"))
    )

    async def scenario() -> None:
        app = create_worker(settings, model, tmp_path / "results.jsonl")
        async with TestRabbitBroker(app.broker) as broker:
            await broker.publish(
                InferenceTask(task_id="batch:1", request=GenerationRequest(prompt="a")),
                queue=task_queue(settings),
            )

    with pytest.raises(OSError, match="disk full"):
        asyncio.run(scenario())
