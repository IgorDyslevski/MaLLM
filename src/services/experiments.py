import json
import logging
import tempfile
import time
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from matplotlib.figure import Figure
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities import Dataset as DatasetEntity
from mlflow.entities import DatasetInput, InputTag, Metric
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.schemas import (
    Dataset,
    DatasetSpec,
    ExperimentPlan,
    ExperimentResult,
    ExperimentRunResult,
    PreparedDataset,
    RunConfig,
    TrainingPoint,
)
from src.services.language_model import TransformerLanguageModel
from src.services.training import train_model
from src.services.training_data import prepare_dataset


def log_metrics(client: MlflowClient, run_id: str, point: TrainingPoint) -> None:
    timestamp = time.time_ns() // 1_000_000
    metrics = [
        Metric(name, float(value), timestamp, point.step)
        for name, value in point.model_dump(exclude={"step"}).items()
    ]
    client.log_batch(run_id, metrics=metrics)


def log_partitions(
    client: MlflowClient, run_id: str, prepared: PreparedDataset, spec: DatasetSpec
) -> None:
    """Зарегистрировать фактически использованные обработанные строки."""
    for partition in (prepared.training, prepared.validation):
        frame = pd.DataFrame(
            [
                {"row_index": str(index), **record.model_dump()}
                for index, record in zip(
                    partition.row_indices, partition.records, strict=True
                )
            ]
        )
        tracked = from_pandas(
            frame,
            source=HTTPDatasetSource(spec.source),
            targets="response",
            name=f"{spec.name}/{partition.context}",
            digest=partition.sha256[:32],
        )
        client.log_inputs(
            run_id,
            datasets=[
                DatasetInput(
                    dataset=DatasetEntity(**tracked.to_dict()),
                    tags=[InputTag("mlflow.data.context", partition.context)],
                )
            ],
        )
        client.log_param(
            run_id, f"dataset.{partition.context}.sha256", partition.sha256
        )
        client.log_param(
            run_id, f"dataset.{partition.context}.rows", len(partition.records)
        )
    client.log_param(run_id, "source.sha256", spec.sha256)
    client.log_param(run_id, "source.revision", spec.revision)
    client.log_param(
        run_id, "preprocessing", "prompt + LF + response; documents joined by two LF"
    )


def execute_run(
    client: MlflowClient,
    run_id: str,
    artifact_uri: str,
    prepared: PreparedDataset,
    spec: DatasetSpec,
    plan: ExperimentPlan,
    variant: RunConfig,
    device: torch.device,
) -> ExperimentRunResult:
    torch.manual_seed(plan.training.seed)
    model = TransformerLanguageModel(variant.model, len(prepared.vocabulary)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    params = {
        **plan.training.model_dump(),
        **variant.model.model_dump(),
        "learning_rate": variant.learning_rate,
        "device": str(device),
        "vocab_size": len(prepared.vocabulary),
        "parameter_count": parameter_count,
        "tokenizer": "character/train-vocabulary/unknown=0",
    }
    for name, value in params.items():
        client.log_param(run_id, name, value)
    log_partitions(client, run_id, prepared, spec)
    result = train_model(
        model,
        prepared,
        variant,
        plan.training,
        device,
        partial(log_metrics, client, run_id),
    )
    client.log_metric(run_id, "training_seconds", result.training_seconds)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "model").mkdir()
        torch.save(
            {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            output / "model" / "weights.pt",
        )
        (output / "model" / "config.json").write_text(
            variant.model.model_dump_json(indent=2), encoding="utf-8"
        )
        (output / "model" / "tokenizer.json").write_text(
            json.dumps({"chars": prepared.vocabulary}, ensure_ascii=False),
            encoding="utf-8",
        )
        (output / "plan.json").write_text(
            plan.model_dump_json(indent=2), encoding="utf-8"
        )
        (output / "dataset_manifest.json").write_text(
            spec.model_dump_json(indent=2), encoding="utf-8"
        )
        (output / "splits.json").write_text(
            json.dumps(
                {
                    "training": prepared.training.row_indices,
                    "validation": prepared.validation.row_indices,
                }
            ),
            encoding="utf-8",
        )
        (output / "history.json").write_text(
            json.dumps([point.model_dump() for point in result.history]),
            encoding="utf-8",
        )
        (output / "sample.txt").write_text(result.sample, encoding="utf-8")
        figure = Figure(figsize=(8, 4), layout="constrained")
        axes = figure.subplots()
        steps = [point.step for point in result.history]
        axes.plot(steps, [point.train_loss for point in result.history], label="train")
        axes.plot(
            steps, [point.val_loss for point in result.history], label="validation"
        )
        axes.set(
            xlabel="Optimizer steps",
            ylabel="Cross-entropy (nats/token)",
            title=variant.name,
        )
        axes.legend()
        figure.savefig(output / "loss.png", dpi=150)
        client.log_artifacts(run_id, directory)
    return ExperimentRunResult(
        name=variant.name,
        run_id=run_id,
        artifact_uri=artifact_uri,
        parameter_count=parameter_count,
        training_seconds=result.training_seconds,
        metrics=result.history[-1],
    )


def log_comparison(
    client: MlflowClient, result: ExperimentResult, plan: ExperimentPlan
) -> None:
    lines = [
        "# Сравнение Transformer",
        f"Общие seed={plan.training.seed}, {plan.training.steps} шагов; "
        f"train={result.training_rows}, validation={result.validation_rows} "
        "документов.",
        "| Run | Val loss | Perplexity | Token accuracy | Parameters | Seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {run.name} | {run.metrics.val_loss:.4f} | "
            f"{run.metrics.val_perplexity:.3f} | {run.metrics.val_accuracy:.2%} | "
            f"{run.parameter_count} | {run.training_seconds:.1f} |"
            for run in result.runs
        ],
        "Loss и accuracy оценивают следующий символ на одинаковых фиксированных "
        f"батчах ({result.runs[0].metrics.val_tokens} токенов validation). "
        "Это короткое сравнение при одном seed, а не оценка точности решения задач.",
    ]
    with tempfile.TemporaryDirectory() as directory:
        figure = Figure(figsize=(9, 4), layout="constrained")
        axes = figure.subplots()
        bars = axes.bar(
            [run.name for run in result.runs],
            [run.metrics.val_loss for run in result.runs],
        )
        axes.bar_label(bars, fmt="%.3f", padding=3)
        axes.set(
            ylabel="Validation cross-entropy (lower is better)",
            title="Shared data, seed and training steps",
        )
        figure.savefig(Path(directory) / "comparison.png", dpi=150)
        for run in result.runs:
            client.log_text(
                run.run_id, result.model_dump_json(indent=2), "comparison.json"
            )
            client.log_text(
                run.run_id,
                "\n\n".join(lines[:2])
                + "\n\n"
                + "\n".join(lines[2:-1])
                + "\n\n"
                + lines[-1],
                "comparison.md",
            )
            client.log_artifact(run.run_id, str(Path(directory) / "comparison.png"))


def run_experiments(
    dataset: Dataset,
    plan: ExperimentPlan,
    client: MlflowClient,
    device: torch.device,
    suite_id: str,
    logger: logging.Logger,
) -> ExperimentResult:
    """Выполнить baseline и варианты на одном разбиении и сохранить сравнение."""
    prepared = prepare_dataset(dataset, plan.training)
    experiment = client.get_experiment_by_name(plan.experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(plan.experiment_name)
    )
    results: list[ExperimentRunResult] = []
    for variant in plan.runs:
        run = client.create_run(
            experiment_id,
            tags={
                "mlflow.runName": variant.name,
                "task": "next_token_prediction",
                "suite_id": suite_id,
                "description": variant.description,
            },
        )
        try:
            result = execute_run(
                client,
                run.info.run_id,
                run.info.artifact_uri,
                prepared,
                dataset.spec,
                plan,
                variant,
                device,
            )
        except (MlflowException, RuntimeError, OSError, ValueError) as exc:
            logger.exception("Ошибка запуска %s", variant.name)
            client.set_terminated(run.info.run_id, status="FAILED")
            raise RuntimeError(f"Не удалось выполнить run {variant.name!r}") from exc
        client.set_terminated(run.info.run_id)
        results.append(result)
    result = ExperimentResult(
        experiment_id=experiment_id,
        suite_id=suite_id,
        training_rows=len(prepared.training.records),
        validation_rows=len(prepared.validation.records),
        runs=results,
    )
    log_comparison(client, result, plan)
    return result
