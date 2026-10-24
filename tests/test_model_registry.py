from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from mlflow.tracking import MlflowClient
from pydantic import ValidationError

from src.schemas import (
    ExperimentResult,
    ExperimentRunResult,
    GenerationRequest,
    RegisteredVersion,
    RegistryConfig,
    TokenizerConfig,
    TrainingPoint,
    TransformerConfig,
)
from src.services.language_model import TransformerLanguageModel
from src.services.model_registry import (
    package_model,
    predict_registered_model,
    register_models,
    verify_alias,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def registry_sources(tmp_path: Path) -> tuple[MlflowClient, ExperimentResult]:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment = client.create_experiment(
        "registry-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    runs: list[ExperimentRunResult] = []
    for name, layers in (("baseline", 2), ("compact-2-layers", 1)):
        run = client.create_run(experiment)
        directory = tmp_path / name
        directory.mkdir()
        config = TransformerConfig(block_size=8, n_embd=8, n_head=2, n_layer=layers)
        model = TransformerLanguageModel(config, 5)
        torch.save(model.state_dict(), directory / "weights.pt")
        (directory / "config.json").write_text(config.model_dump_json())
        tokenizer = TokenizerConfig(chars=["�", "a", "b", " ", "\n"])
        (directory / "tokenizer.json").write_text(tokenizer.model_dump_json())
        client.log_artifacts(run.info.run_id, str(directory), "model")
        client.set_terminated(run.info.run_id)
        runs.append(
            ExperimentRunResult(
                name=name,
                run_id=run.info.run_id,
                artifact_uri=run.info.artifact_uri,
                parameter_count=sum(p.numel() for p in model.parameters()),
                training_seconds=0,
                metrics=TrainingPoint(
                    step=1,
                    train_loss=1,
                    val_loss=1,
                    val_accuracy=0.5,
                    val_perplexity=3,
                    val_tokens=8,
                ),
            )
        )
    return client, ExperimentResult(
        experiment_id=experiment,
        suite_id="test",
        training_rows=9,
        validation_rows=1,
        runs=runs,
    )


def test_registry_versions_load_and_alias_switch_rolls_back(
    registry_sources: tuple[MlflowClient, ExperimentResult],
) -> None:
    client, experiments = registry_sources
    config = RegistryConfig(
        example=GenerationRequest(prompt="unknown a", max_new_tokens=2)
    )
    result = register_models(config, experiments, client, "first")
    assert result.champion.version == "1"
    assert result.candidate.version == "2"
    assert [(check.alias, check.version) for check in result.checks] == [
        ("champion", "1"),
        ("candidate", "2"),
        ("demo", "1"),
        ("demo", "2"),
        ("demo", "1"),
    ]
    prediction = predict_registered_model(
        client, "models:/MaLLM@champion", config.example
    )
    assert prediction.text.startswith("unknown a")
    assert len(prediction.text) == len(config.example.prompt) + 2
    first = client.get_model_version("MaLLM", "1")
    assert first.run_id == experiments.runs[0].run_id
    assert first.tags["verification"] == "passed"
    assert client.list_artifacts(first.run_id, "registry/first")[0].path.endswith(
        "report.json"
    )
    repeated = register_models(config, experiments, client, "second")
    assert repeated.champion.version == "3"
    assert repeated.candidate.version == "4"
    assert client.get_model_version("MaLLM", "1").source == first.source
    assert (
        predict_registered_model(client, "models:/MaLLM/1", config.example)
        == prediction
    )


def test_registry_rejects_missing_source_before_creating_model() -> None:
    client = Mock(spec=MlflowClient)
    experiments = ExperimentResult(
        experiment_id="e", suite_id="s", training_rows=1, validation_rows=1, runs=[]
    )
    with pytest.raises(ValueError, match="отсутствуют"):
        register_models(RegistryConfig(), experiments, client, "test")
    client.create_registered_model.assert_not_called()


def test_registry_rejects_unfinished_source(
    registry_sources: tuple[MlflowClient, ExperimentResult],
    tmp_path: Path,
) -> None:
    client, experiments = registry_sources
    source = experiments.runs[0]
    client.set_terminated(source.run_id, status="FAILED")
    with pytest.raises(ValueError, match="завершённый"):
        package_model(client, source, tmp_path / "model", GenerationRequest(prompt="a"))


def test_registry_detects_alias_pointing_at_different_run() -> None:
    client = Mock(spec=MlflowClient)
    client.get_model_version_by_alias.return_value = Mock(version="2", run_id="other")
    expected = RegisteredVersion(
        version="1", run_id="r", variant="v", model_uri="models:/MaLLM/1"
    )
    with pytest.raises(RuntimeError, match="другую версию"):
        verify_alias(client, RegistryConfig(), "champion", expected)


def test_registry_contract_rejects_reserved_alias_and_duplicate_runs() -> None:
    with pytest.raises(ValidationError, match="рабочих"):
        RegistryConfig(demo_alias="champion")
    with pytest.raises(ValidationError, match="разных Run"):
        RegistryConfig(candidate_run="baseline")
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="", max_new_tokens=0)


def test_registry_cli_is_quiet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.register_model import main

    result = ExperimentResult(
        experiment_id="e", suite_id="s", training_rows=1, validation_rows=1, runs=[]
    )
    results = tmp_path / "results.json"
    results.write_text(result.model_dump_json())
    config = tmp_path / "registry.json"
    config.write_text(RegistryConfig(results_file=results).model_dump_json())
    register = Mock()
    monkeypatch.setattr("src.services.model_registry.register_models", register)
    monkeypatch.setattr("mlflow.tracking.MlflowClient", Mock())
    monkeypatch.setattr("src.logs.setup_logging", Mock())
    monkeypatch.setattr("sys.argv", ["registry", "--config", str(config)])
    main()
    assert register.call_args.args[1] == result
    assert capsys.readouterr().out == ""
