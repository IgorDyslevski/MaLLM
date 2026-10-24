import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import ValidationError

from src.schemas import Dataset, DatasetRecord, DatasetSpec
from src.services.dataset_tracking import log_dataset_input, register_dataset
from src.services.datasets import load_dataset
from src.track_dataset import main


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    records = [DatasetRecord(prompt="What is 2 + 2?", response="4", text="2 + 2 = 4")]
    path = tmp_path / "data.jsonl"
    path.write_text(records[0].model_dump_json() + "\n", encoding="utf-8")
    spec = DatasetSpec(
        name="fixture",
        revision="a" * 40,
        source="https://huggingface.co/datasets/test/fixture/resolve/"
        f"{'a' * 40}/data.jsonl",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_rows=1,
        path=path,
    )
    return Dataset(spec=spec, records=records)


def test_load_dataset_reads_verified_file_without_network(
    dataset: Dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = Mock(side_effect=AssertionError("Network is unavailable"))
    monkeypatch.setattr("src.services.datasets.urlretrieve", download)
    assert load_dataset(dataset.spec) == dataset
    download.assert_not_called()


def test_load_dataset_rejects_changed_bytes(dataset: Dataset) -> None:
    dataset.spec.path.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="Контрольная сумма локального"):
        load_dataset(dataset.spec)


def test_load_dataset_rejects_incomplete_file(dataset: Dataset) -> None:
    dataset.spec.expected_rows = 2
    with pytest.raises(ValueError, match="Количество строк"):
        load_dataset(dataset.spec)


def test_load_dataset_downloads_and_verifies_missing_file(
    dataset: Dataset,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = dataset.spec.path.read_bytes()
    dataset.spec.path = tmp_path / "download" / "data.jsonl"

    def download(url: str, path: Path) -> None:
        assert url == dataset.spec.source
        path.write_bytes(content)

    monkeypatch.setattr("src.services.datasets.urlretrieve", download)
    assert load_dataset(dataset.spec).records == dataset.records


def test_load_dataset_does_not_promote_corrupted_download(
    dataset: Dataset,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset.spec.path = tmp_path / "missing.jsonl"

    def download(url: str, path: Path) -> None:
        path.write_text("corrupt", encoding="utf-8")

    monkeypatch.setattr("src.services.datasets.urlretrieve", download)
    with pytest.raises(ValueError, match="Контрольная сумма скачанного"):
        load_dataset(dataset.spec)
    assert not dataset.spec.path.exists()


def test_dataset_schema_rejects_invalid_hash_and_empty_records(
    dataset: Dataset,
) -> None:
    payload = dataset.spec.model_dump()
    payload["sha256"] = "bad"
    with pytest.raises(ValidationError):
        DatasetSpec.model_validate(payload)
    with pytest.raises(ValidationError):
        Dataset(spec=dataset.spec, records=[])
    with pytest.raises(ValidationError):
        DatasetRecord.model_validate({"prompt": "question", "response": "answer"})


def test_log_dataset_input_preserves_source_digest_schema_and_profile(
    dataset: Dataset,
) -> None:
    client = Mock(spec=MlflowClient)
    log_dataset_input(dataset, client, "run-1")
    logged = client.log_inputs.call_args.kwargs["datasets"][0]
    assert client.log_inputs.call_args.args == ("run-1",)
    assert logged.dataset.name == "fixture/train"
    assert logged.dataset.digest == dataset.spec.sha256[:32]
    assert json.loads(logged.dataset.source)["url"] == dataset.spec.source
    assert json.loads(logged.dataset.profile)["num_rows"] == 1
    assert {
        column["name"] for column in json.loads(logged.dataset.schema)["mlflow_colspec"]
    } == {"prompt", "response", "text"}
    assert logged.tags[0].value == "dataset_tracking"
    client.log_param.assert_any_call("run-1", "dataset.sha256", dataset.spec.sha256)
    manifest = json.loads(client.log_text.call_args.args[1])
    assert manifest["revision"] == dataset.spec.revision
    assert client.log_text.call_args.args[2] == "dataset/manifest.json"


def test_register_dataset_creates_experiment_and_finishes_run(
    dataset: Dataset,
) -> None:
    client = Mock(spec=MlflowClient)
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = "experiment-1"
    client.create_run.return_value.info.run_id = "run-1"
    client.create_run.return_value.info.artifact_uri = "mlflow-artifacts:/run-1"
    result = register_dataset(dataset, client, "dataset-tracking")
    assert result.run_id == "run-1"
    assert result.experiment_id == "experiment-1"
    assert result.dataset_digest == dataset.spec.sha256[:32]
    client.create_experiment.assert_called_once_with("dataset-tracking")
    client.set_terminated.assert_called_once_with("run-1")


def test_register_dataset_reuses_existing_experiment(dataset: Dataset) -> None:
    client = Mock(spec=MlflowClient)
    client.get_experiment_by_name.return_value.experiment_id = "existing"
    client.create_run.return_value.info.run_id = "run-1"
    client.create_run.return_value.info.artifact_uri = "mlflow-artifacts:/run-1"
    assert register_dataset(dataset, client, "existing").experiment_id == "existing"
    client.create_experiment.assert_not_called()


def test_register_dataset_marks_failed_run_and_preserves_cause(
    dataset: Dataset,
) -> None:
    client = Mock(spec=MlflowClient)
    client.create_run.return_value.info.run_id = "run-1"
    client.log_inputs.side_effect = MlflowException("unavailable")
    with pytest.raises(RuntimeError, match="Не удалось зарегистрировать") as error:
        register_dataset(dataset, client, "dataset-tracking")
    assert isinstance(error.value.__cause__, MlflowException)
    client.set_terminated.assert_called_once_with("run-1", status="FAILED")


def test_tracking_cli_loads_manifest_and_prints_registered_run(
    dataset: Dataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(dataset.spec.model_dump_json(), encoding="utf-8")
    client = Mock(spec=MlflowClient)
    client.get_experiment_by_name.return_value.experiment_id = "experiment-1"
    client.create_run.return_value.info.run_id = "run-1"
    client.create_run.return_value.info.artifact_uri = "mlflow-artifacts:/run-1"
    factory = Mock(return_value=client)
    monkeypatch.setattr("src.track_dataset.MlflowClient", factory)
    monkeypatch.setattr(
        "sys.argv",
        [
            "track_dataset",
            "--manifest",
            str(manifest),
            "--tracking-uri",
            "http://tracking.test:5000",
            "--experiment",
            "fixture",
        ],
    )
    main()
    factory.assert_called_once_with(tracking_uri="http://tracking.test:5000")
    client.get_experiment_by_name.assert_called_once_with("fixture")
    assert json.loads(capsys.readouterr().out)["run_id"] == "run-1"
