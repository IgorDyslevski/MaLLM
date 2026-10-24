from pathlib import Path

import pandas as pd
import torch
from mlflow.pyfunc import PythonModel, PythonModelContext

from src.schemas import (
    GenerationRequest,
    GenerationResult,
    TokenizerConfig,
    TransformerConfig,
)
from src.services.language_model import TransformerLanguageModel


class RegisteredLanguageModel(PythonModel):
    """Загрузить Transformer и продолжить переданные тексты."""

    model: TransformerLanguageModel
    vocabulary: list[str]

    def load_context(self, context: PythonModelContext) -> None:
        config = TransformerConfig.model_validate_json(
            Path(context.artifacts["config"]).read_text(encoding="utf-8")
        )
        tokenizer = TokenizerConfig.model_validate_json(
            Path(context.artifacts["tokenizer"]).read_text(encoding="utf-8")
        )
        self.vocabulary = tokenizer.chars
        self.model = TransformerLanguageModel(config, len(self.vocabulary)).eval()
        weights = torch.load(
            context.artifacts["weights"], map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(weights)

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        requests = [
            GenerationRequest.model_validate(row)
            for row in model_input.to_dict("records")
        ]
        outputs = [self.generate(request).model_dump() for request in requests]
        return pd.DataFrame(outputs)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        mapping = {character: index for index, character in enumerate(self.vocabulary)}
        context = torch.tensor(
            [[mapping.get(character, 0) for character in request.prompt]],
            dtype=torch.long,
        )
        generated = self.model.generate(
            context, request.max_new_tokens, torch.Generator().manual_seed(request.seed)
        )
        continuation = "".join(
            self.vocabulary[index] for index in generated[0, context.size(1) :].tolist()
        )
        return GenerationResult(text=request.prompt + continuation)
