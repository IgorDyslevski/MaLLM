"""Проверить остановку воркера, повторную доставку и DLQ на отдельной очереди."""

import asyncio
import json
import os
from asyncio.subprocess import Process
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic
from uuid import uuid4

from aio_pika import RobustQueue
from faststream.rabbit import RabbitQueue
from psycopg import AsyncConnection

from scripts.benchmark import running_process
from src.config import PipelineSettings, Settings
from src.schemas import GenerationRequest, InferenceTask
from src.services.pipeline import create_broker, publish_once, task_queue
from src.services.task_store import connect


async def saved_ids(conn: AsyncConnection, prefix: str) -> list[str]:
    sql = "SELECT task_id FROM inference_results WHERE starts_with(task_id, %s)"
    rows = await conn.execute(sql, (prefix + ":",))
    return [row[0] for row in await rows.fetchall()]


async def wait_ready(
    queue: RobustQueue, count: int, child: Process | None = None
) -> None:
    async with asyncio.timeout(60 if child else 15):
        while (await queue.declare()).message_count != count:
            if child and child.returncode is not None:
                raise ChildProcessError(f"Worker exited with code {child.returncode}")
            await asyncio.sleep(0.05)


@asynccontextmanager
async def worker(queue: RobustQueue, label: str) -> AsyncIterator[Process]:
    """При prefetch=1 дождаться выдачи последней задачи. На выходе послать SIGTERM."""
    directory = Path("artifacts/hw4") / queue.name
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "PIPELINE_QUEUE": queue.name,
        "PIPELINE_QUEUE_CAPACITY": "2",
        "PIPELINE_PREFETCH_COUNT": "1",
        "PIPELINE_INFERENCE_SECONDS": "2",
    }
    async with running_process(
        directory / f"{label}.log", env, "src.inference_worker"
    ) as proc:
        await wait_ready(queue, 0, proc)
        await asyncio.sleep(0.25)  # Дать воркеру войти в двухсекундный расчёт.
        yield proc


async def main() -> None:
    key = f"recovery-{uuid4().hex}"
    settings = PipelineSettings(queue=key, queue_capacity=2)
    request = GenerationRequest(prompt="2+", max_new_tokens=2)
    tasks = [
        InferenceTask(task_id=f"{key}:{suffix}", request=request)
        for suffix in ("term", "waiting-0", "kill", "waiting-1")
    ]
    normal, waiting, interrupted, last = tasks
    database = Settings().database
    async with create_broker(settings) as broker, await connect(database) as db:
        await db.set_autocommit(True)
        route = task_queue(settings)
        queue = await broker.declare_queue(route)
        dlq = await broker.declare_queue(RabbitQueue(key + ".dlq", durable=True))

        await publish_once(normal, broker, settings)
        async with worker(queue, "graceful") as first:
            for task in (waiting, interrupted):
                await publish_once(task, broker, settings)
            assert (await queue.declare()).message_count == 2
            assert not await saved_ids(db, key)
            started = monotonic()
        elapsed = monotonic() - started
        ready_after_sigterm = (await queue.declare()).message_count
        assert first.returncode == 0
        assert ready_after_sigterm == 2
        assert await saved_ids(db, key) == [normal.task_id]
        print(f"SIGTERM: committed before exit, shutdown {elapsed:.2f}s", flush=True)

        async with worker(queue, "crash") as second:
            second.kill()
            await second.wait()
        await wait_ready(queue, 1)
        assert set(await saved_ids(db, key)) == {normal.task_id, waiting.task_id}

        await publish_once(last, broker, settings)
        async with worker(queue, "recovered"):
            assert interrupted.task_id in await saved_ids(db, key)
            print("SIGKILL: unacked redelivered and committed", flush=True)
            await publish_once(interrupted, broker, settings)
            poison = b"invalid JSON"
            await broker.publish(poison, queue=route, message_id=key, persist=True)
            await wait_ready(dlq, 1)
            item = await dlq.get(timeout=2)
            assert item.body == poison
            assert item.headers["attempts"] == 3
            await item.ack()

        saved = await saved_ids(db, key)
        remaining = (await queue.declare()).message_count
        expected = sorted(task.task_id for task in tasks)
        assert remaining == 0
        assert sorted(saved) == expected
        report = {
            "duplicate_results": len(saved) - len(set(saved)),
            "dlq_attempts": item.headers["attempts"],
            "ready_after_sigterm": ready_after_sigterm,
            "valid_tasks": len(tasks),
            "saved_results": len(saved),
            "work_queue_ready": remaining,
        }
        print(json.dumps(report), flush=True)
        for checked in (queue, dlq):
            await checked.delete(if_unused=True, if_empty=True)


if __name__ == "__main__":
    asyncio.run(main())
