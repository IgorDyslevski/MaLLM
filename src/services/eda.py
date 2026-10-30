from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib.figure import Figure
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.schemas import OperationDistribution


def log_distribution(
    distribution: OperationDistribution, client: MlflowClient, run_id: str
) -> None:
    """Сохранить доли операций из карточки датасета и единственный график."""
    figure = Figure(figsize=(10, 5), layout="constrained")
    axis = figure.subplots()
    bars = axis.bar(list(distribution.shares), list(distribution.shares.values()))
    axis.bar_label(bars, fmt="%.0f%%", padding=3)
    axis.set_title("Operation distribution reported in the dataset card")
    axis.set_ylabel("Share, %")
    axis.tick_params(axis="x", rotation=20)
    axis.margins(y=0.15)
    for operation, share in distribution.shares.items():
        client.log_metric(run_id, f"share_percent.{operation}", share)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "operation_distribution.png"
        figure.savefig(path, dpi=150)
        client.log_artifact(run_id, str(path))


def run_eda(
    distribution: OperationDistribution, client: MlflowClient, experiment_name: str
) -> str:
    """Создать Run с распределением из описания датасета."""
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(experiment_name)
    )
    run = client.create_run(
        experiment_id,
        tags={
            "mlflow.runName": "operation-distribution",
            "task": "eda",
            "distribution_source": "dataset_card",
            "source_url": "https://huggingface.co/datasets/nkthebass/math-synth-400k",
        },
    )
    try:
        log_distribution(distribution, client, run.info.run_id)
    except MlflowException as exc:
        client.set_terminated(run.info.run_id, status="FAILED")
        raise RuntimeError("Не удалось сохранить распределение в MLflow") from exc
    client.set_terminated(run.info.run_id)
    return run.info.run_id
