import argparse
import logging
from pathlib import Path
from uuid import uuid4

import torch
from mlflow.tracking import MlflowClient

from src.logs import setup_logging
from src.schemas import DatasetSpec, ExperimentPlan
from src.services.datasets import load_dataset
from src.services.experiments import run_experiments


def main() -> None:
    """Запустить сравнение моделей по конфигурации эксперимента."""
    parser = argparse.ArgumentParser(description="MLflow Experiments & Runs")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.json"))
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/experiments/results.json")
    )
    args = parser.parse_args()
    plan = ExperimentPlan.model_validate_json(args.config.read_text(encoding="utf-8"))
    spec = DatasetSpec.model_validate_json(
        plan.dataset_manifest.read_text(encoding="utf-8")
    )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads)
    setup_logging("WARNING")
    result = run_experiments(
        load_dataset(spec),
        plan,
        MlflowClient(tracking_uri=args.tracking_uri),
        torch.device(device),
        uuid4().hex,
        logging.getLogger(__name__),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
