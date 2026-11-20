import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from mlflow.tracking import MlflowClient

from src.config import PipelineSettings, Settings
from src.logs import setup_logging
from src.services.worker import INFERENCE_SLOTS, initialize_worker


async def run(settings: PipelineSettings) -> None:
    client = MlflowClient(
        tracking_uri=settings.tracking_uri, registry_uri=settings.tracking_uri
    )
    with ThreadPoolExecutor(max_workers=INFERENCE_SLOTS) as pool:
        asyncio.get_running_loop().set_default_executor(pool)
        worker = await initialize_worker(settings, client, Settings().database)
        await worker.run()


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    setup_logging("INFO")
    asyncio.run(run(PipelineSettings()))
