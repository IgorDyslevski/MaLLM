import argparse
import logging
import os


def main() -> None:
    """Сохранить распределение операций датасета в MLflow."""
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "true"
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"

    from mlflow.tracking import MlflowClient

    from src.logs import setup_logging
    from src.schemas import OperationDistribution
    from src.services.eda import run_eda

    parser = argparse.ArgumentParser(description="MLflow EDA: operation distribution")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="mallm-math-eda")
    args = parser.parse_args()
    setup_logging("ERROR")
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    distribution = OperationDistribution(
        shares={
            "addition": 24,
            "subtraction": 20,
            "multiplication": 16,
            "division": 12,
            "order_of_operations": 12,
            "word_problem": 16,
        }
    )
    run_eda(
        distribution,
        MlflowClient(tracking_uri=args.tracking_uri),
        args.experiment,
    )


if __name__ == "__main__":
    main()
