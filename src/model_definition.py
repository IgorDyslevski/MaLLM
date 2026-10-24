from mlflow.models import set_model

from src.services.registered_model import RegisteredLanguageModel

set_model(RegisteredLanguageModel())
