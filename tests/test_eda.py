from pathlib import Path
from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import ValidationError

from src.schemas import OperationDistribution
from src.services.eda import run_eda


def test_eda_saves_only_operation_shares_and_one_chart() -> None:
    client = Mock(spec=MlflowClient)
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = "e"
    client.create_run.return_value.info.run_id = "r"
    distribution = OperationDistribution(shares={"addition": 60, "division": 40})
    images: list[bytes] = []

    def capture(run_id: str, path: str) -> None:
        assert run_id == "r"
        assert Path(path).name == "operation_distribution.png"
        images.append(Path(path).read_bytes())

    client.log_artifact.side_effect = capture
    assert run_eda(distribution, client, "test-eda") == "r"
    client.create_experiment.assert_called_once_with("test-eda")
    client.set_terminated.assert_called_once_with("r")
    assert images[0].startswith(b"\x89PNG")
    client.log_artifact.assert_called_once()
    client.log_text.assert_not_called()
    client.log_metric.assert_any_call("r", "share_percent.addition", 60)
    client.log_metric.assert_any_call("r", "share_percent.division", 40)
    assert client.log_metric.call_count == 2
    assert (
        client.create_run.call_args.kwargs["tags"]["distribution_source"]
        == "dataset_card"
    )


def test_eda_marks_run_failed_and_preserves_mlflow_error() -> None:
    client = Mock(spec=MlflowClient)
    client.get_experiment_by_name.return_value.experiment_id = "e"
    client.create_run.return_value.info.run_id = "r"
    error = MlflowException("unavailable")
    client.log_metric.side_effect = error
    with pytest.raises(RuntimeError, match="распределение") as caught:
        run_eda(OperationDistribution(shares={"addition": 100}), client, "test-eda")
    assert caught.value.__cause__ is error
    client.set_terminated.assert_called_once_with("r", status="FAILED")
    client.create_experiment.assert_not_called()


@pytest.mark.parametrize("share", [-1, 101])
def test_distribution_rejects_share_outside_percent_range(share: float) -> None:
    with pytest.raises(ValidationError):
        OperationDistribution(shares={"addition": share})


def test_eda_cli_uses_documented_shares_without_loading_dataset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.eda import main

    run = Mock()
    client = Mock()
    monkeypatch.setattr("src.services.eda.run_eda", run)
    monkeypatch.setattr("mlflow.tracking.MlflowClient", Mock(return_value=client))
    monkeypatch.setattr("src.logs.setup_logging", Mock())
    monkeypatch.setattr("sys.argv", ["eda"])
    main()
    distribution, actual_client, experiment = run.call_args.args
    assert actual_client == client and experiment == "mallm-math-eda"
    assert distribution.shares == {
        "addition": 24,
        "subtraction": 20,
        "multiplication": 16,
        "division": 12,
        "order_of_operations": 12,
        "word_problem": 16,
    }
    assert sum(distribution.shares.values()) == 100
    assert capsys.readouterr().out == ""
