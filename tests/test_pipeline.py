import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest
from aio_pika.exceptions import DeliveryError
from faststream.rabbit import RabbitBroker, TestRabbitBroker
from faststream.rabbit.message import RabbitMessage
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient
from pamqp.commands import Basic
from psycopg import OperationalError
from pydantic import ValidationError

from src.config import PipelineSettings, Settings
from src.exceptions import InferenceError
from src.schemas import (
    GenerationRequest,
    GenerationResult,
    InferenceTask,
)
from src.services.pipeline import (
    publish_file,
    publish_task,
    task_queue,
)
from src.services.worker import create_worker, execute_task, initialize_worker


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


@pytest.mark.parametrize(
    "prediction_end,finished,delay",
    [
        pytest.param(10.3, 11.0, 0.7, id="fast-model-waits"),
        pytest.param(12.0, 12.0, 0.0, id="slow-model-needs-no-delay"),
    ],
)
def test_execute_task_waits_only_for_remaining_duration(
    monkeypatch: pytest.MonkeyPatch,
    prediction_end: float,
    finished: float,
    delay: float,
) -> None:
    clock = Mock(side_effect=[10.0, prediction_end, finished])
    monkeypatch.setattr("src.services.worker.monotonic", clock)
    sleep = AsyncMock()
    monkeypatch.setattr("src.services.worker.asyncio.sleep", sleep)
    model = Mock(spec=PyFuncModel)
    model.predict.return_value = pd.DataFrame([{"text": "b"}])
    task = InferenceTask(task_id="batch:1", request=GenerationRequest(prompt="a"))
    result = asyncio.run(execute_task(task, model, 1.0))
    sleep.assert_awaited_once_with(pytest.approx(delay, rel=1e-6, abs=0))
    assert result.processing_seconds == finished - 10.0


def test_execute_task_propagates_inference_error() -> None:
    model = Mock(spec=PyFuncModel)
    model.predict.side_effect = RuntimeError("failed")
    task = InferenceTask(task_id="batch:1", request=GenerationRequest(prompt="a"))
    with pytest.raises(InferenceError):
        asyncio.run(execute_task(task, model, 0.001))


def test_worker_loads_alias_once_and_saves_two_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = PipelineSettings(inference_seconds=0.001, _env_file=None)
    model = Mock(spec=PyFuncModel)
    model.predict.return_value = pd.DataFrame([{"text": "b"}])
    loader = Mock(return_value=model)
    save = AsyncMock()
    monkeypatch.setattr("src.services.worker.load_registered_model", loader)
    monkeypatch.setattr("src.services.worker.save_result", save)
    client = Mock(spec=MlflowClient)

    async def scenario() -> None:
        app = await initialize_worker(settings, client, Settings().database)
        async with TestRabbitBroker(app.broker) as broker:
            for number in range(2):
                await broker.publish(
                    InferenceTask(
                        task_id=f"batch:{number}", request=GenerationRequest(prompt="a")
                    ),
                    queue=task_queue(settings),
                )

    asyncio.run(scenario())
    loader.assert_called_once_with(client, settings.model_uri)
    assert model.predict.call_count == 2
    assert [call.args[1].task_id for call in save.await_args_list] == [
        "batch:0",
        "batch:1",
    ]


def test_worker_startup_fails_before_consuming_when_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = Mock(side_effect=RuntimeError("registry unavailable"))
    create = Mock()
    monkeypatch.setattr("src.services.worker.load_registered_model", loader)
    monkeypatch.setattr("src.services.worker.create_worker", create)
    with pytest.raises(RuntimeError, match="registry unavailable"):
        asyncio.run(initialize_worker(PipelineSettings(), Mock(), Settings().database))
    create.assert_not_called()


def test_worker_requeues_delivery_when_result_cannot_be_saved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = PipelineSettings(inference_seconds=0.001, _env_file=None)
    database = Settings().database
    model = Mock(spec=PyFuncModel)
    model.predict.return_value = pd.DataFrame([{"text": "b"}])
    save = AsyncMock(side_effect=OperationalError("write failed"))
    failures = AsyncMock(return_value=1)
    monkeypatch.setattr("src.services.worker.save_result", save)
    monkeypatch.setattr("src.services.worker.record_failure", failures)
    task = InferenceTask(task_id="batch:1", request=GenerationRequest(prompt="a"))

    async def scenario() -> None:
        app = create_worker(settings, model, database)
        async with TestRabbitBroker(app.broker) as broker:
            await broker.publish(
                task, queue=task_queue(settings), message_id=task.task_id
            )

    with patch.object(
        RabbitMessage, "nack", autospec=True, side_effect=RabbitMessage.nack
    ) as nack:
        asyncio.run(scenario())
    save.assert_awaited_once()
    failures.assert_awaited_once_with(database, task.task_id)
    nack.assert_awaited_once()
    assert nack.await_args.kwargs == {"requeue": True}


def test_poison_message_after_max_attempts_goes_to_dlq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = PipelineSettings(_env_file=None)
    monkeypatch.setattr("src.services.worker.record_failure", AsyncMock(return_value=3))
    app = create_worker(settings, Mock(), Settings().database)
    received = []

    @app.broker.subscriber(f"{settings.queue}.dlq")
    async def inspect(body: bytes) -> None:
        received.append(body)

    async def scenario() -> None:
        async with TestRabbitBroker(app.broker) as broker:
            await broker.publish(b"broken", queue=task_queue(settings))

    asyncio.run(scenario())
    assert received == [b"broken"]


def test_event_loop_remains_responsive_during_prediction() -> None:
    model = Mock(spec=PyFuncModel)
    release = threading.Event()

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()

        def predict(frame: pd.DataFrame) -> pd.DataFrame:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(2), "event loop blocked inside model.predict"
            return pd.DataFrame([{"text": "ok"}])

        model.predict.side_effect = predict
        task = InferenceTask(task_id="loop:1", request=GenerationRequest(prompt="a"))
        work = asyncio.create_task(execute_task(task, model, 0.001))
        await asyncio.wait_for(entered.wait(), 2)
        assert not work.done()
        release.set()
        assert (await work).outcome.text == "ok"

    asyncio.run(scenario())
