import argparse
import asyncio
from pathlib import Path
from uuid import uuid4

from src.config import PipelineSettings
from src.logs import setup_logging
from src.services.pipeline import create_broker, publish_file


async def run(source: Path, run_id: str, settings: PipelineSettings) -> None:
    """Отправить файл в очередь и вывести статистику публикации."""
    async with create_broker(settings) as broker:
        result = await publish_file(source, run_id, broker, settings)
    print(result.model_dump_json())


def main() -> None:
    """Запустить источник пакетных задач генерации."""
    parser = argparse.ArgumentParser(
        description="Publish generation requests from JSONL"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-id", default=uuid4().hex)
    args = parser.parse_args()
    setup_logging("INFO")
    asyncio.run(run(args.input, args.run_id, PipelineSettings()))


if __name__ == "__main__":
    main()
