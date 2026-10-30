import asyncio
import logging
from pathlib import Path
from time import monotonic

from aio_pika.exceptions import DeliveryError
from faststream import AckPolicy, FastStream
from faststream.rabbit import Channel, RabbitBroker, RabbitQueue
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient
from pamqp.commands import Basic

from src.config import PipelineSettings
from src.exceptions import InferenceError
from src.schemas import (
    BatchResult,
    GenerationRequest,
    GenerationResult,
    InferenceErrorResponse,
    InferenceTask,
    TaskResult,
)
from src.services.inference import process
from src.services.model_registry import load_registered_model


def task_queue(settings: PipelineSettings) -> RabbitQueue:
    """Описать ограниченную очередь без удаления старых задач."""
    return RabbitQueue(
        settings.queue,
        durable=True,
        arguments={
            "x-max-length": settings.queue_capacity,
            "x-overflow": "reject-publish",
        },
    )


def create_broker(settings: PipelineSettings) -> RabbitBroker:
    """Создать брокер с подтверждением публикаций и одной задачей в работе."""
    return RabbitBroker(
        settings.rabbit_url,
        timeout=10,
        default_channel=Channel(
            prefetch_count=1, publisher_confirms=True, on_return_raises=True
        ),
    )


async def publish_task(
    task: InferenceTask, broker: RabbitBroker, settings: PipelineSettings
) -> None:
    """Дождаться приёма задачи, при переполнении повторить текущую публикацию."""
    while True:
        try:
            await broker.publish(
                task,
                queue=task_queue(settings),
                persist=True,
                message_id=task.task_id,
                correlation_id=task.task_id,
            )
        except DeliveryError as exc:
            if not isinstance(exc.frame, Basic.Nack):
                raise RuntimeError("RabbitMQ could not route the task") from exc
            await asyncio.sleep(settings.retry_seconds)
        else:
            return


async def publish_file(
    source: Path, run_id: str, broker: RabbitBroker, settings: PipelineSettings
) -> BatchResult:
    """Опубликовать JSONL построчно без накопления задач в памяти."""
    started = monotonic()
    published = 0
    await broker.declare_queue(task_queue(settings))
    with source.open(encoding="utf-8") as records:
        for line_number, line in enumerate(records, start=1):
            request = GenerationRequest.model_validate_json(line)
            task = InferenceTask(task_id=f"{run_id}:{line_number}", request=request)
            await publish_task(task, broker, settings)
            published += 1
            await asyncio.sleep(settings.source_interval_seconds)
    return BatchResult(
        run_id=run_id, published=published, elapsed_seconds=monotonic() - started
    )


async def execute_task(
    task: InferenceTask, model: PyFuncModel, inference_seconds: float
) -> TaskResult:
    """Выполнить inference за время не меньше заданной длительности."""
    started = monotonic()
    outcome: GenerationResult | InferenceErrorResponse
    try:
        outcome = await asyncio.to_thread(process, task.request, model)
    except InferenceError as exc:
        logging.getLogger(__name__).exception(
            "Inference failed task_id=%s", task.task_id, exc_info=exc
        )
        outcome = InferenceErrorResponse()
    await asyncio.sleep(max(0, inference_seconds - (monotonic() - started)))
    return TaskResult(
        task_id=task.task_id,
        outcome=outcome,
        processing_seconds=monotonic() - started,
    )


def append_result(destination: Path, result: TaskResult) -> None:
    """Сохранить результат перед подтверждением задачи."""
    with destination.open("a", encoding="utf-8") as results:
        results.write(result.model_dump_json() + "\n")


def create_worker(
    settings: PipelineSettings, model: PyFuncModel, destination: Path
) -> FastStream:
    """Собрать воркер с уже загруженной моделью и сохранением результатов."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    broker = create_broker(settings)

    @broker.subscriber(
        task_queue(settings), ack_policy=AckPolicy.NACK_ON_ERROR, no_reply=True
    )
    async def handle(task: InferenceTask) -> None:
        result = await execute_task(task, model, settings.inference_seconds)
        await asyncio.to_thread(append_result, destination, result)
        logging.getLogger(__name__).info(
            "Task completed task_id=%s processing_seconds=%.3f",
            task.task_id,
            result.processing_seconds,
        )

    return FastStream(broker)


async def initialize_worker(
    settings: PipelineSettings, client: MlflowClient, destination: Path
) -> FastStream:
    """Однократно загрузить модель по alias до подключения потребителя."""
    model = await asyncio.to_thread(load_registered_model, client, settings.model_uri)
    return create_worker(settings, model, destination)
