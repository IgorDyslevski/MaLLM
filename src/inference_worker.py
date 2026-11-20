import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from mlflow.tracking import MlflowClient

from src.config import PipelineSettings, Settings
from src.logs import setup_logging
from src.services.worker import INFERENCE_SLOTS, initialize_worker

logger = logging.getLogger(__name__)


async def run(settings: PipelineSettings) -> None:
    application = Settings()
    setup_logging(application.log_level, "worker", application.environment)
    client = MlflowClient(
        tracking_uri=settings.tracking_uri, registry_uri=settings.tracking_uri
    )
    with ThreadPoolExecutor(max_workers=INFERENCE_SLOTS) as pool:
        asyncio.get_running_loop().set_default_executor(pool)
        worker = await initialize_worker(settings, client, application.database)
        await worker.run()


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "true")
    setup_logging(
        os.getenv("LOG_LEVEL", "INFO"),
        "worker",
        os.getenv("ENVIRONMENT", "development"),
    )
    settings = PipelineSettings()
    try:
        asyncio.run(run(settings))
    except Exception as exc:
        logger.exception("Worker failed", extra={"fields": {"event": "worker_failed"}})
        raise SystemExit(1) from exc
