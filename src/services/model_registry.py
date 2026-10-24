import json
import tempfile
from importlib.metadata import version as package_version
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.models import ModelSignature
from mlflow.pyfunc import PyFuncModel, PythonModelContext
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, Schema

from src.schemas import (
    AliasCheck,
    ExperimentResult,
    ExperimentRunResult,
    GenerationRequest,
    GenerationResult,
    RegisteredVersion,
    RegistryConfig,
    RegistryResult,
)
from src.services.inference import process
from src.services.registered_model import RegisteredLanguageModel


def package_model(
    client: MlflowClient,
    source: ExperimentRunResult,
    destination: Path,
    example: GenerationRequest,
) -> None:
    """Упаковать веса и исходный код в MLflow Model и проверить предсказание."""
    run = client.get_run(source.run_id)
    if run.info.status != "FINISHED":
        raise ValueError("Для регистрации нужен успешно завершённый Run")
    artifacts: dict[str, str] = {}
    for name, filename in (
        ("weights", "weights.pt"),
        ("config", "config.json"),
        ("tokenizer", "tokenizer.json"),
    ):
        artifacts[name] = client.download_artifacts(
            source.run_id, f"model/{filename}", str(destination.parent)
        )
    reference = RegisteredLanguageModel()
    reference.load_context(PythonModelContext(artifacts=artifacts, model_config=None))
    inputs = pd.DataFrame([example.model_dump()])
    expected = reference.predict(inputs)
    root = Path(__file__).resolve().parents[1]
    requirements = [
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        *[
            f"{name}=={package_version(name)}"
            for name in ("torch", "mlflow", "pandas", "pydantic")
        ],
    ]
    signature = ModelSignature(
        inputs=Schema(
            [
                ColSpec("string", "prompt"),
                ColSpec("long", "max_new_tokens", required=False),
                ColSpec("long", "seed", required=False),
            ]
        ),
        outputs=Schema([ColSpec("string", "text")]),
    )
    mlflow.pyfunc.save_model(
        path=str(destination),
        python_model=str(root / "model_definition.py"),
        code_paths=[str(root)],
        artifacts=artifacts,
        signature=signature,
        input_example=inputs,
        pip_requirements=requirements,
    )
    restored = mlflow.pyfunc.load_model(str(destination))
    if not restored.predict(inputs).equals(expected):
        raise RuntimeError("MLflow Model не воспроизводит предсказание исходной модели")


def register_version(
    client: MlflowClient,
    config: RegistryConfig,
    source: ExperimentRunResult,
    registration_id: str,
) -> RegisteredVersion:
    """Создать версию из отдельного неизменяемого пакета исходного Run."""
    artifact_path = f"registry_models/{registration_id}"
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "mlflow_model"
        package_model(client, source, destination, config.example)
        client.log_artifacts(source.run_id, str(destination), artifact_path)
    registered = client.create_model_version(
        config.name,
        source=f"runs:/{source.run_id}/{artifact_path}",
        run_id=source.run_id,
        await_creation_for=30,
        tags={
            "variant": source.name,
            "val_loss": str(source.metrics.val_loss),
            "verification": "passed",
        },
        description=(
            f"{source.name}; validation loss={source.metrics.val_loss:.4f}; "
            f"source run={source.run_id}"
        ),
    )
    if registered.status != "READY":
        raise RuntimeError("Версия модели не готова к использованию")
    return RegisteredVersion(
        version=str(registered.version),
        run_id=source.run_id,
        variant=source.name,
        model_uri=f"models:/{config.name}/{registered.version}",
    )


def load_registered_model(client: MlflowClient, model_uri: str) -> PyFuncModel:
    """Загрузить модель по версии или alias через заданный MLflow-клиент."""
    if not model_uri.startswith("models:/"):
        raise ValueError("Ожидается URI models:/name/version или models:/name@alias")
    selection = model_uri.removeprefix("models:/")
    if "@" in selection:
        name, alias = selection.split("@", 1)
        version = str(client.get_model_version_by_alias(name, alias).version)
    else:
        name, version = selection.split("/", 1)
    source = client.get_model_version_download_uri(name, version)
    with tempfile.TemporaryDirectory() as directory:
        local = download_artifacts(
            artifact_uri=source,
            dst_path=directory,
            tracking_uri=client.tracking_uri,
            registry_uri=client.tracking_uri,
        )
        return mlflow.pyfunc.load_model(local)


def predict_registered_model(
    client: MlflowClient, model_uri: str, request: GenerationRequest
) -> GenerationResult:
    """Загрузить модель по версии или alias и продолжить текст."""
    return process(request, load_registered_model(client, model_uri))


def verify_alias(
    client: MlflowClient,
    config: RegistryConfig,
    alias: str,
    expected: RegisteredVersion,
) -> AliasCheck:
    """Проверить разрешение alias и совпадение с неизменяемой версией."""
    resolved = client.get_model_version_by_alias(config.name, alias)
    if str(resolved.version) != expected.version or resolved.run_id != expected.run_id:
        raise RuntimeError("Alias указывает на другую версию модели")
    uri = f"models:/{config.name}@{alias}"
    prediction = predict_registered_model(client, uri, config.example)
    reference = predict_registered_model(client, expected.model_uri, config.example)
    if prediction != reference:
        raise RuntimeError("Предсказание через alias не совпадает с выбранной версией")
    return AliasCheck(
        alias=alias, version=expected.version, model_uri=uri, prediction=prediction.text
    )


def register_models(
    config: RegistryConfig,
    experiments: ExperimentResult,
    client: MlflowClient,
    registration_id: str,
) -> RegistryResult:
    """Зарегистрировать версии и продемонстрировать переключение и откат alias."""
    runs = {run.name: run for run in experiments.runs}
    if config.champion_run not in runs or config.candidate_run not in runs:
        raise ValueError("В результатах эксперимента отсутствуют выбранные Runs")
    existing = client.search_registered_models(
        filter_string=f"name = '{config.name}'", max_results=1
    )
    if not existing:
        client.create_registered_model(
            config.name,
            description=(
                "Символьный Transformer для математических текстов; "
                "версии связаны с обучающими Runs."
            ),
        )
    champion = register_version(
        client, config, runs[config.champion_run], registration_id
    )
    candidate = register_version(
        client, config, runs[config.candidate_run], registration_id
    )
    client.set_registered_model_alias(config.name, "champion", champion.version)
    client.set_registered_model_alias(config.name, "candidate", candidate.version)
    checks = [
        verify_alias(client, config, "champion", champion),
        verify_alias(client, config, "candidate", candidate),
    ]
    for selected in (champion, candidate, champion):
        client.set_registered_model_alias(
            config.name, config.demo_alias, selected.version
        )
        checks.append(verify_alias(client, config, config.demo_alias, selected))
    result = RegistryResult(
        name=config.name, champion=champion, candidate=candidate, checks=checks
    )
    report = {
        "registration_id": registration_id,
        **result.model_dump(),
    }
    for selected in (champion, candidate):
        client.log_text(
            selected.run_id,
            json.dumps(report, ensure_ascii=False, indent=2),
            f"registry/{registration_id}/report.json",
        )
    return result
