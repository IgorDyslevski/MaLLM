import argparse
import logging
import os
from pathlib import Path
from uuid import uuid4


def main() -> None:
    """Зарегистрировать готовые модели и проверить версии и alias без обучения."""
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "true"
    os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "false"
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "false"

    import torch
    from mlflow.tracking import MlflowClient

    from src.logs import setup_logging
    from src.schemas import ExperimentResult, RegistryConfig
    from src.services.model_registry import register_models

    parser = argparse.ArgumentParser(
        description="MLflow Model Registry: versions and aliases"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/registry.json"))
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    config = RegistryConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    experiments = ExperimentResult.model_validate_json(
        config.results_file.read_text(encoding="utf-8")
    )
    torch.set_num_threads(1)
    setup_logging("ERROR")
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    register_models(
        config,
        experiments,
        MlflowClient(tracking_uri=args.tracking_uri, registry_uri=args.tracking_uri),
        uuid4().hex,
    )


if __name__ == "__main__":
    main()
