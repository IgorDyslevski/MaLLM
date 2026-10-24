from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class VersionResponse(BaseModel):
    version: str = Field(min_length=1)


class ComponentHealth(BaseModel):
    status: Literal["ok", "error"]
    message: str
    version: str | None = None
    response_time_ms: float = Field(ge=0)


class DependencyHealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: dict[str, ComponentHealth]


class DatasetRecord(BaseModel):
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DatasetSpec(BaseModel):
    name: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source: str = Field(pattern=r"^https://huggingface.co/datasets/")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_rows: int = Field(gt=0)
    split: Literal["train"] = "train"
    path: Path


class Dataset(BaseModel):
    spec: DatasetSpec
    records: list[DatasetRecord] = Field(min_length=1)


class DatasetTrackingResult(BaseModel):
    run_id: str
    experiment_id: str
    dataset_digest: str
    artifact_uri: str


class TransformerConfig(BaseModel):
    block_size: int = Field(default=128, gt=0)
    n_embd: int = Field(default=64, gt=0)
    n_head: int = Field(default=4, gt=0)
    n_layer: int = Field(default=4, gt=0)
    dropout: float = Field(default=0, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_heads(self) -> Self:
        if self.n_embd % self.n_head:
            raise ValueError("Размер эмбеддинга должен делиться на число голов")
        return self


class RunConfig(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    model: TransformerConfig = Field(default_factory=TransformerConfig)
    learning_rate: float = Field(default=1e-3, gt=0)


class TrainingConfig(BaseModel):
    seed: int = Field(default=42, ge=0)
    max_records: int = Field(default=20000, ge=2)
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)
    steps: int = Field(default=500, gt=0)
    batch_size: int = Field(default=16, gt=0)
    eval_interval: int = Field(default=100, gt=0)
    eval_batches: int = Field(default=20, gt=0)
    final_eval_batches: int = Field(default=100, gt=0)


class ExperimentPlan(BaseModel):
    experiment_name: str = Field(min_length=1)
    dataset_manifest: Path
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    runs: list[RunConfig] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_runs(self) -> Self:
        names = [run.name for run in self.runs]
        if names[0] != "baseline" or len(names) != len(set(names)):
            raise ValueError(
                "Первый run должен быть baseline; имена должны быть уникальны"
            )
        variants = {
            run.model_dump_json(exclude={"name", "description"}) for run in self.runs
        }
        if len(variants) != len(self.runs):
            raise ValueError("Runs должны отличаться моделью или гиперпараметрами")
        return self


class DatasetPartition(BaseModel):
    context: Literal["training", "validation"]
    row_indices: list[int] = Field(min_length=1)
    records: list[DatasetRecord] = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PreparedDataset(BaseModel):
    training: DatasetPartition
    validation: DatasetPartition
    vocabulary: list[str] = Field(min_length=2)


class EvaluationMetrics(BaseModel):
    loss: float = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    perplexity: float = Field(ge=1)
    tokens: int = Field(gt=0)


class TrainingPoint(BaseModel):
    step: int = Field(ge=0)
    train_loss: float = Field(ge=0)
    val_loss: float = Field(ge=0)
    val_accuracy: float = Field(ge=0, le=1)
    val_perplexity: float = Field(ge=1)
    val_tokens: int = Field(gt=0)


class TrainingResult(BaseModel):
    history: list[TrainingPoint] = Field(min_length=1)
    training_seconds: float = Field(ge=0)
    sample: str


class ExperimentRunResult(BaseModel):
    name: str
    run_id: str
    artifact_uri: str
    parameter_count: int = Field(gt=0)
    training_seconds: float = Field(ge=0)
    metrics: TrainingPoint


class ExperimentResult(BaseModel):
    experiment_id: str
    suite_id: str
    training_rows: int = Field(gt=0)
    validation_rows: int = Field(gt=0)
    runs: list[ExperimentRunResult]


class TokenizerConfig(BaseModel):
    chars: list[str] = Field(min_length=2)


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=32, ge=1, le=1024)
    seed: int = Field(default=42, ge=0)


class GenerationResult(BaseModel):
    text: str


class InferenceErrorResponse(BaseModel):
    status: Literal["inference_error"] = "inference_error"
    message: str = "Model inference failed"


class RegistryConfig(BaseModel):
    name: str = Field(default="MaLLM", pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    results_file: Path = Path("artifacts/experiments/results.json")
    champion_run: str = "baseline"
    candidate_run: str = "compact-2-layers"
    demo_alias: str = Field(default="demo", pattern=r"^[a-z][a-z0-9_-]*$")
    example: GenerationRequest = Field(
        default_factory=lambda: GenerationRequest(
            prompt="What is 2 + 2?\n", max_new_tokens=8
        )
    )

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.champion_run == self.candidate_run:
            raise ValueError("Для демонстрации версий нужны два разных Run")
        if self.demo_alias in {"champion", "candidate"}:
            raise ValueError("Демонстрационный alias должен отличаться от рабочих")
        return self


class RegisteredVersion(BaseModel):
    version: str
    run_id: str
    variant: str
    model_uri: str


class AliasCheck(BaseModel):
    alias: str
    version: str
    model_uri: str
    prediction: str


class RegistryResult(BaseModel):
    name: str
    champion: RegisteredVersion
    candidate: RegisteredVersion
    checks: list[AliasCheck]
