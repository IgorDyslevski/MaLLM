import argparse
from pathlib import Path

from mlflow.tracking import MlflowClient

from src.schemas import DatasetSpec
from src.services.dataset_tracking import register_dataset
from src.services.datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="MLflow Dataset Tracking")
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/math-synth-400k.json")
    )
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="mallm-dataset-tracking")
    args = parser.parse_args()
    spec = DatasetSpec.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    result = register_dataset(
        load_dataset(spec),
        MlflowClient(tracking_uri=args.tracking_uri),
        args.experiment,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
