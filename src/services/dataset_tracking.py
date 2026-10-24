import pandas as pd
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities import Dataset as DatasetEntity
from mlflow.entities import DatasetInput, InputTag
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.schemas import Dataset, DatasetTrackingResult


def log_dataset_input(dataset: Dataset, client: MlflowClient, run_id: str) -> None:
    frame = pd.DataFrame.from_records([row.model_dump() for row in dataset.records])
    tracked = from_pandas(
        frame,
        source=HTTPDatasetSource(dataset.spec.source),
        name=f"{dataset.spec.name}/{dataset.spec.split}",
        targets="response",
        digest=dataset.spec.sha256[:32],
    )
    client.log_inputs(
        run_id,
        datasets=[
            DatasetInput(
                dataset=DatasetEntity(**tracked.to_dict()),
                tags=[InputTag("mlflow.data.context", "dataset_tracking")],
            )
        ],
    )
    client.log_param(run_id, "dataset.sha256", dataset.spec.sha256)
    client.log_param(run_id, "dataset.revision", dataset.spec.revision)
    client.log_param(run_id, "dataset.split", dataset.spec.split)
    client.log_param(run_id, "dataset.rows", len(dataset.records))
    client.log_text(
        run_id, dataset.spec.model_dump_json(indent=2), "dataset/manifest.json"
    )


def register_dataset(
    dataset: Dataset, client: MlflowClient, experiment_name: str
) -> DatasetTrackingResult:
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(experiment_name)
    )
    run = client.create_run(
        experiment_id,
        tags={"mlflow.runName": "register-dataset", "task": "dataset_tracking"},
    )
    try:
        log_dataset_input(dataset, client, run.info.run_id)
    except MlflowException as exc:
        client.set_terminated(run.info.run_id, status="FAILED")
        raise RuntimeError("Не удалось зарегистрировать датасет в MLflow") from exc
    client.set_terminated(run.info.run_id)
    return DatasetTrackingResult(
        run_id=run.info.run_id,
        experiment_id=experiment_id,
        dataset_digest=dataset.spec.sha256[:32],
        artifact_uri=run.info.artifact_uri,
    )
