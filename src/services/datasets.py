import hashlib
from urllib.request import urlretrieve

from src.schemas import Dataset, DatasetRecord, DatasetSpec


def load_dataset(spec: DatasetSpec) -> Dataset:
    if not spec.path.exists():
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        partial = spec.path.with_suffix(".part")
        urlretrieve(spec.source, partial)
        with partial.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != spec.sha256:
            raise ValueError("Контрольная сумма скачанного датасета не совпадает")
        partial.replace(spec.path)
    with spec.path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != spec.sha256:
        raise ValueError("Контрольная сумма локального датасета не совпадает")
    with spec.path.open(encoding="utf-8") as stream:
        records = [DatasetRecord.model_validate_json(line) for line in stream]
    if len(records) != spec.expected_rows:
        raise ValueError("Количество строк датасета не совпадает с манифестом")
    return Dataset(spec=spec, records=records)
