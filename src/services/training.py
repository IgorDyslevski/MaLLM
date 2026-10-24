import math
import time
from collections.abc import Callable

import torch
from torch.nn import functional as functional

from src.schemas import (
    EvaluationMetrics,
    PreparedDataset,
    RunConfig,
    TrainingConfig,
    TrainingPoint,
    TrainingResult,
)
from src.services.language_model import TransformerLanguageModel
from src.services.training_data import encode_partition


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Выбрать контексты с целями, сдвинутыми на один токен."""
    if len(data) <= block_size:
        raise ValueError("Число токенов в split должно превышать размер блока")
    starts = torch.randint(len(data) - block_size, (batch_size, 1), generator=generator)
    positions = starts + torch.arange(block_size)
    return data[positions].to(device), data[positions + 1].to(device)


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    )


@torch.no_grad()
def evaluate_model(
    model: TransformerLanguageModel,
    data: torch.Tensor,
    config: TrainingConfig,
    batches: int,
    device: torch.device,
) -> EvaluationMetrics:
    """Оценить одинаковые фиксированные батчи при каждом вызове."""
    generator = torch.Generator().manual_seed(config.seed + 1)
    was_training = model.training
    model.eval()
    loss_sum, correct, tokens = 0.0, 0, 0
    for _ in range(batches):
        inputs, targets = get_batch(
            data, config.batch_size, model.block_size, device, generator
        )
        logits = model(inputs)
        count = targets.numel()
        loss_sum += next_token_loss(logits, targets).item() * count
        correct += int((logits.argmax(dim=-1) == targets).sum())
        tokens += count
    model.train(was_training)
    loss = loss_sum / tokens
    return EvaluationMetrics(
        loss=loss, accuracy=correct / tokens, perplexity=math.exp(loss), tokens=tokens
    )


def evaluate_checkpoint(
    model: TransformerLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: TrainingConfig,
    batches: int,
    device: torch.device,
    step: int,
) -> TrainingPoint:
    training = evaluate_model(model, train_data, config, batches, device)
    validation = evaluate_model(model, val_data, config, batches, device)
    return TrainingPoint(
        step=step,
        train_loss=training.loss,
        val_loss=validation.loss,
        val_accuracy=validation.accuracy,
        val_perplexity=validation.perplexity,
        val_tokens=validation.tokens,
    )


def train_model(
    model: TransformerLanguageModel,
    prepared: PreparedDataset,
    variant: RunConfig,
    config: TrainingConfig,
    device: torch.device,
    record_metrics: Callable[[TrainingPoint], None],
) -> TrainingResult:
    """Обучить один вариант и передать историю метрик в MLflow."""
    train_data = encode_partition(prepared.training, prepared.vocabulary)
    val_data = encode_partition(prepared.validation, prepared.vocabulary)
    optimizer = torch.optim.AdamW(model.parameters(), lr=variant.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    history: list[TrainingPoint] = []
    started = time.perf_counter()
    model.train()
    for step in range(config.steps + 1):
        if step % config.eval_interval == 0 or step == config.steps:
            batches = (
                config.final_eval_batches
                if step == config.steps
                else config.eval_batches
            )
            point = evaluate_checkpoint(
                model, train_data, val_data, config, batches, device, step
            )
            history.append(point)
            record_metrics(point)
        if step == config.steps:
            break
        inputs, targets = get_batch(
            train_data, config.batch_size, model.block_size, device, generator
        )
        loss = next_token_loss(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    training_seconds = time.perf_counter() - started
    model.eval()
    mapping = {char: index for index, char in enumerate(prepared.vocabulary)}
    prompt = "What is 2 + 2?\n"
    context = torch.tensor(
        [[mapping.get(char, 0) for char in prompt]], dtype=torch.long, device=device
    )
    generated = model.generate(
        context, 200, torch.Generator(device=device).manual_seed(config.seed)
    )
    sample = prompt + "".join(
        prepared.vocabulary[index] for index in generated[0, context.size(1) :].tolist()
    )
    return TrainingResult(
        history=history, training_seconds=training_seconds, sample=sample
    )
