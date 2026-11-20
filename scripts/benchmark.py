"""Снять временные ряды на настоящих процессах источника и воркеров."""

import argparse
import asyncio
import csv
import json
import os
import sys
from bisect import bisect_right
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from decimal import Decimal
from pathlib import Path
from statistics import mean, median, quantiles
from time import time
from typing import NamedTuple
from uuid import uuid4

from faststream.rabbit import RabbitQueue

from src.config import PipelineSettings, Settings
from src.services.pipeline import create_broker, task_queue
from src.services.task_store import connect


class Sample(NamedTuple):
    timestamp: float
    ready: int
    completed: int
    source_rss_mib: float
    workers_rss_mib: float


class TaskTiming(NamedTuple):
    completed_at: Decimal
    processing_seconds: float
    end_to_end_seconds: Decimal


def rss_mib(pid: int) -> float:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return 0.0
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return 0.0


@asynccontextmanager
async def running_process(
    log_path: Path, env: dict[str, str], module: str, *args: str
) -> AsyncIterator[asyncio.subprocess.Process]:
    """Запустить модуль с выводом в файл и остановить его при выходе."""
    command = (sys.executable, "-m", module, *args)
    with log_path.open("w") as log:
        child = await asyncio.create_subprocess_exec(
            *command, env=env, stdout=log, stderr=log
        )
    try:
        yield child
    finally:
        if child.returncode is None:
            child.terminate()
            await asyncio.wait_for(child.wait(), 90)


async def measure(
    directory: Path,
    name: str,
    limit: int | None,
    prefetch: int,
    workers: int,
    count: int,
) -> dict[str, str | int | float]:
    directory.mkdir(parents=True)
    run_id = f"bench-{uuid4().hex}"
    env = {
        **os.environ,
        "PIPELINE_QUEUE": run_id,
        "PIPELINE_QUEUE_CAPACITY": str(limit) if limit else "null",
        "PIPELINE_PREFETCH_COUNT": str(prefetch),
        "PIPELINE_SOURCE_INTERVAL_SECONDS": "0",
    }
    settings = PipelineSettings(queue=run_id, queue_capacity=limit)
    database = Settings().database
    confirms = directory / "confirms.csv"
    source_file = directory / "input.jsonl"
    source_file.write_text('{"prompt":"2+2=","max_new_tokens":2}\n' * count)
    samples: list[Sample] = []
    async with create_broker(settings) as broker, await connect(database) as conn:
        queue = await broker.declare_queue(task_queue(settings))
        await conn.set_autocommit(True)
        async with AsyncExitStack() as processes:
            consumers = []
            for i in range(workers):
                log = directory / f"worker-{i}.log"
                worker = running_process(log, env, "src.inference_worker")
                consumers.append(await processes.enter_async_context(worker))
            async with asyncio.timeout(90):
                while (await queue.declare()).consumer_count != workers:
                    if any(w.returncode is not None for w in consumers):
                        raise RuntimeError(f"Worker startup failed: {directory}")
                    await asyncio.sleep(0.2)
            publisher = running_process(
                directory / "source.log",
                env,
                "src.batch_source",
                "--input",
                str(source_file),
                "--run-id",
                run_id,
                "--confirms",
                str(confirms),
            )
            source = await processes.enter_async_context(publisher)
            async with asyncio.timeout(60 + count * settings.inference_seconds * 2):
                while True:
                    if source.returncode not in (None, 0) or any(
                        w.returncode is not None for w in consumers
                    ):
                        raise RuntimeError(f"Process failed: {directory}")
                    ready = (await queue.declare()).message_count
                    cur = await conn.execute(
                        "SELECT count(*) FROM inference_results WHERE task_id LIKE %s",
                        (run_id + ":%",),
                    )
                    done = (await cur.fetchone())[0]
                    samples.append(
                        Sample(
                            timestamp=time(),
                            ready=ready,
                            completed=done,
                            source_rss_mib=rss_mib(source.pid),
                            workers_rss_mib=sum(rss_mib(w.pid) for w in consumers),
                        )
                    )
                    if done == count and source.returncode == 0:
                        break
                    await asyncio.sleep(0.15)
        cur = await conn.execute(
            """SELECT extract(epoch FROM completed_at), processing_seconds,
                      extract(epoch FROM completed_at - submitted_at)
               FROM inference_results WHERE task_id LIKE %s ORDER BY completed_at""",
            (run_id + ":%",),
        )
        timings = [TaskTiming(*row) for row in await cur.fetchall()]
        # Только уникальные очереди этого прогона; if_empty защищает принятые задачи.
        await queue.delete(if_unused=True, if_empty=True)
        dlq = await broker.declare_queue(RabbitQueue(run_id + ".dlq", durable=True))
        await dlq.delete(if_unused=True, if_empty=True)
    confirmations = [
        float(line.split(",")[0]) for line in confirms.read_text().splitlines()
    ]
    if len(confirmations) != count:
        raise RuntimeError("Unexpected number of publisher confirms")
    write_series(directory / "series.csv", samples, confirmations, timings)
    return summarize(name, workers, samples, confirmations, timings)


def write_series(
    path: Path,
    samples: list[Sample],
    confirmations: list[float],
    timings: list[TaskTiming],
) -> None:
    """Записать состояние очереди и последний готовый результат на момент замера."""
    with path.open("w") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "seconds",
                "ready",
                "confirmed",
                "completed",
                "source_rss_mib",
                "workers_rss_mib",
                "latest_processing_s",
                "latest_end_to_end_s",
            )
        )
        for sample in samples:
            latest = timings[sample.completed - 1] if sample.completed else None
            writer.writerow(
                (
                    round(sample.timestamp - confirmations[0], 3),
                    sample.ready,
                    bisect_right(confirmations, sample.timestamp),
                    sample.completed,
                    round(sample.source_rss_mib, 1),
                    round(sample.workers_rss_mib, 1),
                    round(latest.processing_seconds, 3) if latest else "",
                    round(latest.end_to_end_seconds, 3) if latest else "",
                )
            )


def summarize(
    name: str,
    workers: int,
    samples: list[Sample],
    confirmations: list[float],
    timings: list[TaskTiming],
) -> dict[str, str | int | float]:
    count = len(confirmations)
    # Скорость считаем по средней половине задач, как в таблице README.
    first, last = count // 4, count * 3 // 4
    source_seconds = confirmations[last] - confirmations[first]
    worker_seconds = float(timings[last].completed_at - timings[first].completed_at)
    latencies = [task.end_to_end_seconds for task in timings]
    p95 = quantiles(latencies, n=20, method="inclusive")[-1]
    return {
        "scenario": name,
        "tasks": count,
        "workers": workers,
        "ready_max": max(sample.ready for sample in samples),
        "source_per_s": round((last - first) / source_seconds, 2),
        "worker_per_s": round((last - first) / worker_seconds, 2),
        "source_rss_mib": round(max(sample.source_rss_mib for sample in samples), 1),
        "workers_rss_mib": round(max(sample.workers_rss_mib for sample in samples), 1),
        "mean_processing_s": round(
            mean(task.processing_seconds for task in timings), 3
        ),
        "median_latency_s": round(float(median(latencies)), 3),
        "p95_latency_s": round(float(p95), 3),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/hw4") / f"bench-{int(time())}"
    )
    args = parser.parse_args()
    if args.tasks < 8:
        parser.error("At least 8 tasks required")
    summary = []
    # Имя, лимит очереди (None — без лимита), prefetch (0 — без лимита), число воркеров.
    cases = [
        ("unlimited", None, 0, 1),
        ("unbounded-queue", None, 1, 1),
        ("bounded-1", 8, 1, 1),
        ("bounded-2", 8, 1, 2),
    ]
    for name, limit, prefetch, workers in cases:
        result = await measure(
            args.output / name, name, limit, prefetch, workers, args.tasks
        )
        print(json.dumps(result), flush=True)
        summary.append(result)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
