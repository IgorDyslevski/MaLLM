import os


def main() -> None:
    """Запустить эксперименты без информационного вывода MLflow в консоль."""
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "true"
    os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"

    from src.experiments import main as run_cli

    run_cli()


if __name__ == "__main__":
    main()
