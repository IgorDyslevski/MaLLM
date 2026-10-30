import argparse
import asyncio
import os
from pathlib import Path

from mlflow.tracking import MlflowClient

from src.config import PipelineSettings
from src.logs import setup_logging
from src.services.pipeline import initialize_worker


async def run(settings: PipelineSettings, destination: Path) -> None:
    """Загрузить модель один раз и запустить обработку очереди."""
    client = MlflowClient(
        tracking_uri=settings.tracking_uri, registry_uri=settings.tracking_uri
    )
    worker = await initialize_worker(settings, client, destination)
    await worker.run()


def main() -> None:
    """Запустить воркер inference для зарегистрированной модели."""
    parser = argparse.ArgumentParser(
        description="Consume generation tasks from RabbitMQ"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/pipeline/results.jsonl")
    )
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    setup_logging("INFO")
    asyncio.run(run(PipelineSettings(), args.output))


if __name__ == "__main__":
    main()
