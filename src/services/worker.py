import asyncio
import logging
from hashlib import sha256
from time import monotonic

from faststream import AckPolicy, FastStream
from faststream.rabbit import RabbitMessage, RabbitQueue
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient
from psycopg import Error as DatabaseError

from src.config import DatabaseSettings, PipelineSettings
from src.schemas import InferenceTask, TaskResult
from src.services.inference import process
from src.services.model_registry import load_registered_model
from src.services.pipeline import PUBLISH_TIMEOUT_SECONDS, create_broker, task_queue
from src.services.task_store import initialize_tables, record_failure, save_result

logger = logging.getLogger(__name__)
INFERENCE_SLOTS = 1
MAX_ATTEMPTS = 3


async def execute_task(
    task: InferenceTask, model: PyFuncModel, inference_seconds: float
) -> TaskResult:
    started = monotonic()
    outcome = await asyncio.to_thread(process, task.request, model)
    await asyncio.sleep(max(0, inference_seconds - (monotonic() - started)))
    return TaskResult(
        task_id=task.task_id, outcome=outcome, processing_seconds=monotonic() - started
    )


def raw_body(message: RabbitMessage) -> bytes:
    """Отложить валидацию до обработчика, чтобы повреждённый JSON попал в DLQ."""
    return message.raw_message.body


def create_worker(
    settings: PipelineSettings, model: PyFuncModel, database: DatabaseSettings
) -> FastStream:
    broker = create_broker(settings)
    app = FastStream(broker)
    dlq = RabbitQueue(f"{settings.queue}.dlq", durable=True)

    @app.on_startup
    async def prepare() -> None:
        await initialize_tables(database)
        await broker.connect()
        await broker.declare_queue(dlq)

    slots = asyncio.Semaphore(INFERENCE_SLOTS)

    async def process_delivery(body: bytes) -> None:
        task = InferenceTask.model_validate_json(body)
        async with slots:
            result = await execute_task(task, model, settings.inference_seconds)
            await save_result(database, result, task.submitted_at)
        logger.info("Completed task_id=%s", task.task_id)

    @broker.subscriber(
        task_queue(settings),
        ack_policy=AckPolicy.NACK_ON_ERROR,
        decoder=raw_body,
        no_reply=True,
    )
    async def handle(body: bytes, message: RabbitMessage) -> None:
        try:
            await process_delivery(body)
        except Exception as exc:  # noqa: BLE001 -- отказ одной доставки учитывается в БД
            key = message.raw_message.message_id or sha256(body).hexdigest()
            try:
                attempts = await record_failure(database, key)
            except DatabaseError:
                # Пока БД недоступна, доставка остаётся без подтверждения.
                await asyncio.sleep(settings.retry_seconds)
                raise
            logger.warning("Failed message_id=%s attempt=%d", key, attempts)
            if attempts < MAX_ATTEMPTS:
                await message.nack(requeue=True)
                return
            # Сначала получить confirm от DLQ, затем вернуть управление для ack.
            await broker.publish(
                body,
                queue=dlq,
                persist=True,
                mandatory=True,
                message_id=key,
                correlation_id=message.correlation_id,
                headers={"attempts": attempts, "error": type(exc).__name__},
                content_type=message.content_type,
                timeout=PUBLISH_TIMEOUT_SECONDS,
            )

    return app


async def initialize_worker(
    settings: PipelineSettings, client: MlflowClient, database: DatabaseSettings
) -> FastStream:
    model = await asyncio.to_thread(load_registered_model, client, settings.model_uri)
    return create_worker(settings, model, database)
