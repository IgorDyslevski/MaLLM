import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel

from src.exceptions import InferenceError
from src.schemas import GenerationRequest, GenerationResult


def process(request: GenerationRequest, model: PyFuncModel) -> GenerationResult:
    """Продолжить текст уже загруженной моделью."""
    inputs = pd.DataFrame([request.model_dump()])
    try:
        prediction = model.predict(inputs)
    except (MlflowException, RuntimeError, ValueError) as exc:
        raise InferenceError("Model inference failed") from exc
    return GenerationResult.model_validate(prediction.iloc[0].to_dict())
