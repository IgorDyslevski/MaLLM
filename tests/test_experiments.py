import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import ValidationError

from src.experiments import main
from src.schemas import (
    Dataset,
    DatasetRecord,
    DatasetSpec,
    ExperimentPlan,
    RunConfig,
    TrainingConfig,
    TrainingPoint,
    TransformerConfig,
)
from src.services.experiments import run_experiments
from src.services.language_model import Head, TransformerLanguageModel
from src.services.training import (
    evaluate_model,
    get_batch,
    train_model,
)
from src.services.training_data import encode_partition, prepare_dataset


@pytest.fixture(scope="module", autouse=True)
def cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return Dataset(
        spec=DatasetSpec(
            name="fixture",
            revision="a" * 40,
            sha256="b" * 64,
            source="https://huggingface.co/datasets/test/fixture/data.jsonl",
            expected_rows=40,
            path=tmp_path / "data.jsonl",
        ),
        records=[
            DatasetRecord(
                prompt=f"Question {index}",
                response="Answer 4",
                text=f"<|user|>Question {index}<|end|><|assistant|>Answer 4<|end|>",
            )
            for index in range(40)
        ],
    )


@pytest.fixture
def plan(tmp_path: Path) -> ExperimentPlan:
    architecture = TransformerConfig(block_size=8, n_embd=8, n_head=2, n_layer=2)
    return ExperimentPlan(
        experiment_name="test-comparison",
        dataset_manifest=tmp_path / "manifest.json",
        training=TrainingConfig(
            max_records=40,
            steps=2,
            batch_size=2,
            eval_interval=1,
            eval_batches=1,
            final_eval_batches=2,
        ),
        runs=[
            RunConfig(name="baseline", description="baseline", model=architecture),
            RunConfig(
                name="compact",
                description="half depth",
                model=architecture.model_copy(update={"n_layer": 1}),
            ),
            RunConfig(
                name="lower-lr",
                description="lower rate",
                model=architecture,
                learning_rate=3e-4,
            ),
        ],
    )


def make_client() -> Mock:
    client = Mock(spec=MlflowClient)
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = "experiment-1"
    runs = []
    for index in range(3):
        run = Mock()
        run.info.run_id = f"run-{index}"
        run.info.artifact_uri = f"mlflow-artifacts:/run-{index}"
        runs.append(run)
    client.create_run.side_effect = runs
    return client


def test_partition_groups_questions_and_is_reproducible(dataset: Dataset) -> None:
    dataset.records.append(dataset.records[0])
    config = TrainingConfig(max_records=100)
    prepared = prepare_dataset(dataset, config)
    assert prepared == prepare_dataset(dataset, config)
    assert set(prepared.training.row_indices).isdisjoint(
        prepared.validation.row_indices
    )
    assert len(prepared.training.records) + len(prepared.validation.records) == 41
    train_questions = {record.prompt for record in prepared.training.records}
    val_questions = {record.prompt for record in prepared.validation.records}
    assert train_questions.isdisjoint(val_questions)
    assert prepared.training.sha256 != prepared.validation.sha256
    assert all("<|" not in record.text for record in prepared.training.records)


def test_partition_hash_changes_with_content_and_vocabulary_ignores_validation(
    dataset: Dataset,
) -> None:
    config = TrainingConfig(max_records=40)
    before = prepare_dataset(dataset, config)
    dataset.records[before.validation.row_indices[0]].response = "\u2603"
    after = prepare_dataset(dataset, config)
    assert before.training.sha256 == after.training.sha256
    assert before.validation.sha256 != after.validation.sha256
    assert "\u2603" not in after.vocabulary
    assert (encode_partition(after.validation, after.vocabulary) == 0).any()


def test_partition_rejects_single_question(dataset: Dataset) -> None:
    for record in dataset.records:
        record.prompt = "same"
    with pytest.raises(ValueError, match="два разных вопроса"):
        prepare_dataset(dataset, TrainingConfig())


def test_attention_matches_pytorch_reference() -> None:
    config = TransformerConfig(block_size=8, n_embd=8, n_head=2, n_layer=1)
    head = Head(config, 4).eval()
    inputs = torch.randn(2, 6, 8)
    expected = torch.nn.functional.scaled_dot_product_attention(
        head.query(inputs),
        head.key(inputs),
        head.value(inputs),
        is_causal=True,
    )
    torch.testing.assert_close(head(inputs), expected)


def test_transformer_masks_future_and_generates_without_gradients() -> None:
    config = TransformerConfig(block_size=8, n_embd=8, n_head=2, n_layer=1)
    model = TransformerLanguageModel(config, 5).eval()
    tokens = torch.randint(0, 5, (1, 8))
    changed = tokens.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 5
    torch.testing.assert_close(model(tokens)[:, :4], model(changed)[:, :4])
    flags: list[bool] = []

    def capture(
        module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        flags.append(output.requires_grad)

    hook = model.register_forward_hook(capture)
    generated = model.generate(tokens, 3, torch.Generator().manual_seed(1))
    hook.remove()
    assert generated.shape == (1, 11)
    assert torch.equal(generated[:, :8], tokens)
    assert flags == [False, False, False]
    with pytest.raises(ValueError, match="превышает"):
        model(generated)


def test_batches_shift_targets_and_reject_short_data() -> None:
    inputs, targets = get_batch(
        torch.arange(40), 2, 8, torch.device("cpu"), torch.Generator()
    )
    assert torch.equal(targets, inputs + 1)
    with pytest.raises(ValueError, match="превышать"):
        get_batch(torch.arange(8), 2, 8, torch.device("cpu"), torch.Generator())


def test_evaluation_uses_fixed_batches_and_preserves_model_mode(
    plan: ExperimentPlan,
) -> None:
    model = TransformerLanguageModel(plan.runs[0].model, 5)
    data = torch.arange(100) % 5
    before = evaluate_model(model, data, plan.training, 2, torch.device("cpu"))
    after = evaluate_model(model, data, plan.training, 2, torch.device("cpu"))
    assert before == after
    assert before.tokens == 2 * 2 * 8
    assert model.training
    model.eval()
    evaluate_model(model, data, plan.training, 1, torch.device("cpu"))
    assert not model.training


def test_training_reduces_loss_on_tiny_dataset(
    dataset: Dataset, plan: ExperimentPlan
) -> None:
    prepared = prepare_dataset(dataset, plan.training)
    model = TransformerLanguageModel(plan.runs[0].model, len(prepared.vocabulary))
    config = plan.training.model_copy(update={"steps": 60, "eval_interval": 30})
    variant = plan.runs[0].model_copy(update={"learning_rate": 0.02})
    history: list[int] = []

    def capture(point: TrainingPoint) -> None:
        history.append(point.step)

    result = train_model(
        model,
        prepared,
        variant,
        config,
        torch.device("cpu"),
        capture,
    )
    assert result.history[-1].val_loss < result.history[0].val_loss
    assert len(history) == 3
    assert result.sample.startswith("What is 2 + 2?\n")
    assert not model.training


def test_run_comparison_tracks_shared_inputs_metrics_and_reloadable_weights(
    dataset: Dataset,
    plan: ExperimentPlan,
) -> None:
    client = make_client()
    artifacts: dict[str, dict[str, bytes]] = {}

    def capture(run_id: str, directory: str) -> None:
        root = Path(directory)
        artifacts[run_id] = {
            str(file.relative_to(root)): file.read_bytes()
            for file in root.rglob("*")
            if file.is_file()
        }

    client.log_artifacts.side_effect = capture
    result = run_experiments(
        dataset, plan, client, torch.device("cpu"), "suite", logging.getLogger(__name__)
    )
    assert len(result.runs) == 3
    assert result.runs[1].parameter_count < result.runs[0].parameter_count
    assert result.runs[0].metrics.val_tokens == result.runs[1].metrics.val_tokens
    assert result.training_rows + result.validation_rows == 40
    assert client.set_terminated.call_count == 3
    assert client.log_inputs.call_count == 6
    assert client.log_batch.call_count == 9
    inputs = [call.kwargs["datasets"][0] for call in client.log_inputs.call_args_list]
    assert len({entry.dataset.digest for entry in inputs[::2]}) == 1
    assert len({entry.dataset.digest for entry in inputs[1::2]}) == 1
    assert inputs[0].dataset.digest != dataset.spec.sha256[:32]
    assert [entry.tags[0].value for entry in inputs[:2]] == ["training", "validation"]
    for files in artifacts.values():
        assert files["loss.png"].startswith(b"\x89PNG")
        config = TransformerConfig.model_validate_json(files["model/config.json"])
        chars = json.loads(files["model/tokenizer.json"])["chars"]
        restored = TransformerLanguageModel(config, len(chars))
        restored.load_state_dict(
            torch.load(io.BytesIO(files["model/weights.pt"]), weights_only=True)
        )
        assert restored(torch.ones((1, 4), dtype=torch.long)).shape == (
            1,
            4,
            len(chars),
        )
    client.log_text.assert_any_call(
        "run-0", result.model_dump_json(indent=2), "comparison.json"
    )


def test_run_comparison_marks_failed_run_and_preserves_cause(
    dataset: Dataset, plan: ExperimentPlan
) -> None:
    client = make_client()
    client.get_experiment_by_name.return_value = Mock(experiment_id="existing")
    client.log_inputs.side_effect = MlflowException("unavailable")
    with pytest.raises(RuntimeError, match="baseline") as error:
        run_experiments(
            dataset,
            plan,
            client,
            torch.device("cpu"),
            "suite",
            logging.getLogger(__name__),
        )
    assert isinstance(error.value.__cause__, MlflowException)
    client.set_terminated.assert_called_once_with("run-0", status="FAILED")
    client.create_experiment.assert_not_called()


def test_experiment_contract_rejects_invalid_architecture_and_duplicate_variants(
    plan: ExperimentPlan,
) -> None:
    with pytest.raises(ValidationError, match="делиться"):
        TransformerConfig(n_embd=65, n_head=4)
    payload = plan.model_dump()
    payload["runs"][1]["name"] = "baseline"
    with pytest.raises(ValidationError, match="уникальны"):
        ExperimentPlan.model_validate(payload)
    payload = plan.model_dump()
    payload["runs"][1]["model"] = payload["runs"][0]["model"]
    with pytest.raises(ValidationError, match="отличаться"):
        ExperimentPlan.model_validate(payload)


def test_cli_loads_plan_and_saves_comparison(
    dataset: Dataset,
    plan: ExperimentPlan,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.schemas import ExperimentResult

    plan.dataset_manifest.write_text(dataset.spec.model_dump_json(), encoding="utf-8")
    config_path = tmp_path / "experiments.json"
    config_path.write_text(plan.model_dump_json(), encoding="utf-8")
    result = ExperimentResult(
        experiment_id="e", suite_id="s", training_rows=36, validation_rows=4, runs=[]
    )
    output = tmp_path / "out" / "results.json"
    run = Mock(return_value=result)
    monkeypatch.setattr("src.experiments.load_dataset", Mock(return_value=dataset))
    monkeypatch.setattr("src.experiments.MlflowClient", Mock())
    monkeypatch.setattr("src.experiments.run_experiments", run)
    monkeypatch.setattr("src.experiments.setup_logging", Mock())
    monkeypatch.setattr(
        "sys.argv", ["main.py", "--config", str(config_path), "--output", str(output)]
    )
    main()
    assert json.loads(output.read_text())["experiment_id"] == "e"
    assert run.call_args.args[0] == dataset
    assert capsys.readouterr().out == ""


def test_entrypoint_suppresses_mlflow_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    from main import main as entrypoint

    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "false")
    monkeypatch.setenv("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "false")
    run = Mock()
    monkeypatch.setattr("src.experiments.main", run)
    entrypoint()
    assert os.environ["MLFLOW_DISABLE_AGENT_HINT"] == "true"
    assert os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] == "true"
    run.assert_called_once_with()
