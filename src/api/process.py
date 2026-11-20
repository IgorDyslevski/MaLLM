import asyncio
import logging
from math import ceil
from typing import Annotated
from uuid import uuid4

from aio_pika.exceptions import AMQPException
from fastapi import APIRouter, Depends, HTTPException, Request
from faststream.rabbit import RabbitBroker

from src.config import PipelineSettings
from src.logs import task_context
from src.schemas import GenerationRequest, InferenceTask, TaskAccepted
from src.services.pipeline import PUBLISH_TIMEOUT_SECONDS, QueueFullError, publish_once

process_router = APIRouter(tags=["Inference"])
logger = logging.getLogger(__name__)


def current_broker(request: Request) -> RabbitBroker:
    return request.state.broker


def current_pipeline(request: Request) -> PipelineSettings:
    return request.state.pipeline


@process_router.post(
    "/api/v1/tasks",
    status_code=202,
    responses={
        503: {
            "description": "Очередь переполнена или брокер недоступен",
            "headers": {
                "Retry-After": {
                    "description": "Пауза в секундах",
                    "schema": {"type": "integer"},
                }
            },
        }
    },
)
async def submit_task(
    request: GenerationRequest,
    broker: Annotated[RabbitBroker, Depends(current_broker)],
    settings: Annotated[PipelineSettings, Depends(current_pipeline)],
) -> TaskAccepted:
    task = InferenceTask(task_id=uuid4().hex, request=request)
    with task_context(task.task_id):
        try:
            async with asyncio.timeout(PUBLISH_TIMEOUT_SECONDS):
                await publish_once(task, broker, settings)
        except (QueueFullError, AMQPException, OSError, RuntimeError) as exc:
            logger.warning(
                "Task rejected with HTTP 503",
                extra={
                    "fields": {
                        "event": "task_rejected",
                        "status_code": 503,
                        "retry_after": ceil(settings.inference_seconds),
                    }
                },
            )
            raise HTTPException(
                503,
                detail="Task queue unavailable",
                headers={"Retry-After": str(ceil(settings.inference_seconds))},
            ) from exc
        logger.info("Task accepted", extra={"fields": {"event": "task_accepted"}})
    return TaskAccepted(task_id=task.task_id)
