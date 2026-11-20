import asyncio
from collections.abc import Callable
from pathlib import Path
from time import monotonic

from aio_pika.exceptions import DeliveryError
from faststream.rabbit import Channel, RabbitBroker, RabbitQueue
from pamqp.commands import Basic

from src.config import PipelineSettings
from src.schemas import BatchResult, GenerationRequest, InferenceTask

PUBLISH_TIMEOUT_SECONDS = 2.0


class QueueFullError(Exception):
    """Брокер отклонил публикацию в заполненную очередь."""


def task_queue(settings: PipelineSettings) -> RabbitQueue:
    arguments: dict[str, str | int] = {"x-queue-type": "classic"}
    if settings.queue_capacity is not None:
        arguments.update(
            {"x-max-length": settings.queue_capacity, "x-overflow": "reject-publish"}
        )
    return RabbitQueue(settings.queue, durable=True, arguments=arguments)


def create_broker(settings: PipelineSettings) -> RabbitBroker:
    return RabbitBroker(
        settings.rabbit_url,
        timeout=10,
        graceful_timeout=max(60, settings.inference_seconds + 30),
        default_channel=Channel(
            prefetch_count=settings.prefetch_count,
            publisher_confirms=True,
            on_return_raises=True,
        ),
    )


async def publish_once(
    task: InferenceTask, broker: RabbitBroker, settings: PipelineSettings
) -> None:
    try:
        await broker.publish(
            task,
            queue=task_queue(settings),
            persist=True,
            mandatory=True,
            timeout=PUBLISH_TIMEOUT_SECONDS,
            message_id=task.task_id,
            correlation_id=task.task_id,
        )
    except DeliveryError as exc:
        if isinstance(exc.frame, Basic.Nack):
            raise QueueFullError from exc
        raise RuntimeError("RabbitMQ could not route the task") from exc


async def publish_task(
    task: InferenceTask, broker: RabbitBroker, settings: PipelineSettings
) -> None:
    while True:
        try:
            await publish_once(task, broker, settings)
            return
        except QueueFullError:
            await asyncio.sleep(settings.retry_seconds)


async def publish_file(
    source: Path,
    run_id: str,
    broker: RabbitBroker,
    settings: PipelineSettings,
    on_confirm: Callable[[int], None] | None = None,
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
            if on_confirm is not None:
                on_confirm(published)
            if settings.source_interval_seconds:
                await asyncio.sleep(settings.source_interval_seconds)
    return BatchResult(
        run_id=run_id, published=published, elapsed_seconds=monotonic() - started
    )
