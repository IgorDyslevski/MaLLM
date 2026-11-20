import asyncio
from pathlib import Path

import pytest
import yaml
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from src.config import PipelineSettings
from src.schemas import GenerationRequest, InferenceTask
from src.services.pipeline import (
    QueueFullError,
    create_broker,
    publish_once,
    publish_task,
    task_queue,
)


def test_rabbit_rejects_full_queue_and_source_resumes_after_consume() -> None:
    compose = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
    )
    container = DockerContainer(compose["services"]["rabbitmq"]["image"])
    container.ports["5672/tcp"] = ("127.0.0.1", None)
    container.waiting_for(
        LogMessageWaitStrategy("Server startup complete").with_startup_timeout(60)
    )

    async def scenario(settings: PipelineSettings) -> None:
        first, second = [
            InferenceTask(task_id=str(i), request=GenerationRequest(prompt="a"))
            for i in range(2)
        ]
        async with create_broker(settings) as broker:
            queue = await broker.declare_queue(task_queue(settings))
            await publish_once(first, broker, settings)
            with pytest.raises(QueueFullError):
                await publish_once(second, broker, settings)
            pending = asyncio.create_task(publish_task(second, broker, settings))
            await asyncio.sleep(0.2)
            assert not pending.done()
            saved = await queue.get(timeout=2)
            assert saved.message_id == first.task_id
            await saved.ack()
            await asyncio.wait_for(pending, timeout=3)
            saved = await queue.get(timeout=2)
            assert saved.message_id == second.task_id
            await saved.ack()

    with container:
        port = container.get_exposed_port(5672)
        settings = PipelineSettings(
            rabbit_url=f"amqp://guest:guest@127.0.0.1:{port}/",
            queue_capacity=1,
            _env_file=None,
        )
        asyncio.run(scenario(settings))
