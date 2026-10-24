import hashlib
import random
from typing import Literal

import torch

from src.schemas import (
    Dataset,
    DatasetPartition,
    PreparedDataset,
    TrainingConfig,
)


def make_partition(
    dataset: Dataset, indices: list[int], context: Literal["training", "validation"]
) -> DatasetPartition:
    records = [
        dataset.records[index].model_copy(
            update={
                "text": f"{dataset.records[index].prompt}\n"
                f"{dataset.records[index].response}"
            }
        )
        for index in indices
    ]
    digest = hashlib.sha256()
    for index, record in zip(indices, records, strict=True):
        digest.update(f"{index}:{record.model_dump_json()}\n".encode())
    return DatasetPartition(
        context=context, row_indices=indices, records=records, sha256=digest.hexdigest()
    )


def prepare_dataset(dataset: Dataset, config: TrainingConfig) -> PreparedDataset:
    """Выбрать общие данные и разделить вопросы до построения словаря."""
    rng = random.Random(config.seed)
    selected = sorted(
        rng.sample(
            range(len(dataset.records)), min(config.max_records, len(dataset.records))
        )
    )
    questions = sorted({dataset.records[index].prompt for index in selected})
    rng.shuffle(questions)
    split_at = int((1 - config.validation_fraction) * len(questions))
    if split_at == 0 or split_at == len(questions):
        raise ValueError("Для train/validation нужны хотя бы два разных вопроса")
    train_questions = set(questions[:split_at])
    train_indices = [
        index for index in selected if dataset.records[index].prompt in train_questions
    ]
    val_indices = [
        index
        for index in selected
        if dataset.records[index].prompt not in train_questions
    ]
    training = make_partition(dataset, train_indices, "training")
    validation = make_partition(dataset, val_indices, "validation")
    characters = set("\n\n".join(record.text for record in training.records))
    return PreparedDataset(
        training=training,
        validation=validation,
        vocabulary=["\ufffd", *sorted(characters - {"\ufffd"})],
    )


def encode_partition(
    partition: DatasetPartition, vocabulary: list[str]
) -> torch.Tensor:
    """Закодировать документы, разделённые двумя переводами строки."""
    mapping = {character: index for index, character in enumerate(vocabulary)}
    text = "\n\n".join(record.text for record in partition.records)
    return torch.tensor(
        [mapping.get(character, 0) for character in text], dtype=torch.long
    )
